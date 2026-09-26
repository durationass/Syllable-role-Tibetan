import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import glob
from soundfile import read, write
from tqdm import tqdm
from pesq import pesq
import torch
import time
from argparse import ArgumentParser
from os.path import join
import pandas as pd
import re
from geco.data_module import SpecsDataModule
from geco.sdes import BBED
from fastgeco.model import ScoreModel
from geco.util.other import pad_spec
from pesq import pesq
# from wvmos import get_wvmos
from pystoi import stoi
import os
import torchaudio
from utils import print_mean_std, si_sdr
import shutil
from pathlib import Path

from function_script.eval.deterministic_sampling import SAMPLING_POLICY, utterance_rng

try:
    from ptflops import get_model_complexity_info
except ImportError:
    get_model_complexity_info = None


def round_metric(value):
    if pd.isna(value):
        return value
    value_at_three_decimals = Decimal(f"{float(value):.3f}")
    return float(value_at_three_decimals.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def format_metric(value):
    if pd.isna(value):
        return "nan"
    return f"{round_metric(value):.2f}"


def mean_std_half_up(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return "nan ± nan"
    mean = round_metric(np.mean(values))
    std = round_metric(np.std(values))
    return f"{mean:.2f} ± {std:.2f}"


def read_text_table(path):
    last_error = None
    for encoding in ("utf-8-sig", "utf-16", "gb18030"):
        try:
            with open(path, "r", encoding=encoding) as f:
                rows = {}
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if "\t" in line:
                        key, text = line.split("\t", 1)
                    else:
                        parts = line.split(maxsplit=1)
                        if len(parts) != 2:
                            continue
                        key, text = parts
                    rows[key.strip()] = text.strip()
                return rows
        except UnicodeError as e:
            last_error = e
    raise UnicodeError(f"Could not decode {path}: {last_error}")


def speech_path_to_utt_id(path):
    stem = Path(str(path).replace("\\", "/")).stem
    parent = Path(str(path).replace("\\", "/")).parent.name
    return f"{parent}_{stem}"


def load_tibetan_references(test_dir, label_file, metadata_file=""):
    if not label_file:
        return {}
    metadata_file = metadata_file or os.path.join(test_dir, "metadata.csv")
    labels = read_text_table(label_file)
    metadata = pd.read_csv(metadata_file)
    refs = {}
    missing = 0
    for _, row in metadata.iterrows():
        sample_id = str(row["id"])
        utt_id = speech_path_to_utt_id(row["speech_path"])
        text = labels.get(utt_id, "")
        if not text:
            missing += 1
        refs[sample_id] = {"utt_id": utt_id, "ref_text": text}
    if missing:
        print(f"Warning: {missing} metadata rows had no matching Tibetan transcript.")
    return refs


if __name__ == '__main__':
    def str2bool(v):
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"}

    parser = ArgumentParser()
    parser.add_argument("--type", type=str, default='test', help="Name of destination folder")
    parser.add_argument("--destination_folder", type=str, default='',
                        help="Alias of --type for compatibility.")
    parser.add_argument("--output_dir", type=str, default='',
                        help="Direct output directory. If set, type/destination_folder is ignored.")
    parser.add_argument("--test_dir", type=str, default='./zang_data_sepformer/test', help='Directory containing the test data')
    parser.add_argument("--ckpt", type=str, default='./logs/z1u7rmzd/epoch=7-si_sdr=9.44.ckpt', help='Path to model checkpoint.')
    parser.add_argument("--reverse_starting_point", type=float, default=None, help="Override reverse SDE starting point.")
    parser.add_argument(
        "--correction_alpha",
        type=float,
        default=1.0,
        help="Interpolate Fast-GeCo output with frontend prediction: y + alpha * (x_hat - y).",
    )
    parser.add_argument("--debug", nargs='?', const=True, default=False, type=str2bool,
                        help="Quick check mode: only run 2 files. Supports '--debug' or '--debug true/false'.")
    parser.add_argument("--max_index", type=int, default=400,
                        help="Only evaluate files whose numeric index is <= max_index. Use 0 to disable this filter.")
    parser.add_argument("--max_files", type=int, default=0,
                        help="Maximum number of test files to evaluate after index filtering. Use 0 to run all selected files.")
    parser.add_argument("--compute_gmacs", nargs='?', const=True, default=True, type=str2bool,
                        help="Compute model GMACs with ptflops. Disable via --compute_gmacs false.")
    parser.add_argument("--save_audio", nargs='?', const=True, default=True, type=str2bool,
                        help="Save enhanced wavs and copied ref/mix/pred files. Disable via --save_audio false.")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Random seed for deterministic reverse sampling.")
    parser.add_argument("--tibetan_label_file", type=str, default="",
                        help="Optional Tibetan label.txt. If set, eval writes transcript fields and an ASR manifest.")
    parser.add_argument("--tibetan_metadata_file", type=str, default="",
                        help="Optional metadata.csv. Defaults to <test_dir>/metadata.csv when Tibetan labels are enabled.")

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    mixture_files = sorted(glob.glob(os.path.join(args.test_dir, '*_mix.wav')))

    # 修改：正则表达式支持可选负号（如 -3601）
    if args.max_index and args.max_index > 0:
        filtered_files = []
        for f in mixture_files:
            base = os.path.basename(f)
            # 允许文件名以负号开头，捕获后面的数字部分
            match = re.search(r"-?(\d+)_mix\.wav$", base)
            if match and int(match.group(1)) <= args.max_index:
                filtered_files.append(f)
        mixture_files = filtered_files

    noisy_files = [item.replace('_mix.wav', '_source1hatP.wav') for item in mixture_files]
    clean_files = [item.replace('_mix.wav', '_source1.wav') for item in mixture_files]
    
    if args.debug:
        limit = 2
    elif args.max_files and args.max_files > 0:
        limit = args.max_files
    else:
        limit = None

    if limit is not None:
        clean_files = clean_files[:limit]
        noisy_files = noisy_files[:limit]
        mixture_files = mixture_files[:limit]

    print(f"Eval files: {len(mixture_files)}")
    if len(mixture_files) == 0:
        print("Warning: No mixture files found. Possible reasons:")
        print("  1. The path --test_dir does not contain '*_mix.wav' files.")
        print("  2. The numeric index in filenames may exceed --max_index (current {})".format(args.max_index))
        print("  3. Try adding '--max_index 0' to disable index filtering.")
        print("  4. Check if your filenames contain unexpected characters (e.g., leading '-').")
        raise SystemExit(1)
    
    # wvmos_model = get_wvmos(cuda=True)
    checkpoint_file = args.ckpt

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    # Detect Git LFS pointer files early to provide a clear fix.
    with open(checkpoint_file, "rb") as f:
        header = f.read(64)
    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            "Checkpoint appears to be a Git LFS pointer, not real model weights. "
            "Run 'git lfs pull' (or download the real .ckpt file) and retry."
        )

    folder_name = args.destination_folder if args.destination_folder else args.type
    target_dir = args.output_dir if args.output_dir else "./Libri2mix/{}/".format(folder_name)

    os.makedirs(target_dir, exist_ok=True)
    if args.save_audio:
        os.makedirs(os.path.join(target_dir, "files"), exist_ok=True)

    tibetan_refs = load_tibetan_references(
        args.test_dir, args.tibetan_label_file, args.tibetan_metadata_file
    )
    if tibetan_refs and not args.save_audio:
        print("Warning: Tibetan ASR manifest is most useful with --save_audio true.")

    # Load score model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    inference_config = checkpoint.get("inference_config", {})
    model = ScoreModel.load_from_checkpoint(
        checkpoint_file,
        batch_size=16, num_workers=0, kwargs=dict(gpu=False)
    )
    model.eval(no_ema=False)
    model.to(device)
    t_eps = float(getattr(model, "t_eps", 0.03))

    # Settings: default to the checkpoint validation strategy, allow explicit CLI overrides.
    reverse_starting_point = (
        args.reverse_starting_point
        if args.reverse_starting_point is not None
        else float(inference_config.get("inference_start", getattr(model, "inference_start", 0.5)))
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    gmacs = None
    if args.compute_gmacs:
        if get_model_complexity_info is None:
            print("ptflops 未安装，跳过 GMACs 统计。可执行: pip install ptflops")
        else:
            try:
                freq_bins = int(getattr(model.data_module, "n_fft", 510) // 2 + 1)
                time_frames = int(getattr(model.data_module, "num_frames", 256))

                def input_constructor(_):
                    x = torch.randn(1, 6, freq_bins, time_frames, device=device)
                    t = torch.ones(1, device=device) * reverse_starting_point
                    divide_scale = t[:, None, None, None]
                    return {
                        "x": x,
                        "time_cond": t,
                        "scale_divide": divide_scale,
                    }

                macs, _ = get_model_complexity_info(
                    model.dnn,
                    (6, freq_bins, time_frames),
                    input_constructor=input_constructor,
                    as_strings=False,
                    print_per_layer_stat=False,
                    verbose=False,
                )
                gmacs = macs / 1e9
            except Exception as e:
                print(f"GMACs 统计失败，已跳过: {e}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    sr = 8000
    data = {
        "filename": [], "pesq": [], "estoi": [], "si_sdr": [],
        "latency_s": [], "audio_len_s": [], "rtf": [], "sampling_seed": []
    }
    asr_manifest = []
    for clean_file, noisy_file, mixture_file in tqdm(
        zip(clean_files, noisy_files, mixture_files), total=len(mixture_files)
    ):
        
        filename = os.path.basename(noisy_file)
        sample_id = (
            filename[:-len("_source1hatP.wav")]
            if filename.endswith("_source1hatP.wav")
            else os.path.splitext(filename)[0]
        )
        # Load wav
        x, sr_ = torchaudio.load(clean_file)
        if sr_ != sr:
            x = torchaudio.transforms.Resample(sr_, sr)(x)
        y, sr_ = torchaudio.load(noisy_file)
        if sr_ != sr:
            y = torchaudio.transforms.Resample(sr_, sr)(y)
        m, sr_ = torchaudio.load(mixture_file)
        if sr_ != sr:
            m = torchaudio.transforms.Resample(sr_, sr)(m)
            
        min_leng = min(x.shape[-1],y.shape[-1],m.shape[-1])
        x = x[...,:min_leng]
        y = y[...,:min_leng]
        m = m[...,:min_leng]
        T_orig = x.size(1)   

        # Normalize per utterance
        norm_factor = y.abs().max().clamp_min(1e-8)
        y = y / norm_factor
        x = x / norm_factor
        m = m / norm_factor 
        
        noise = y - x

        # Prepare DNN input
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.to(device))), 0)
        Y = pad_spec(Y)
        
        X = torch.unsqueeze(model._forward_transform(model._stft(x.to(device))), 0)
        X = pad_spec(X)
        
        M = torch.unsqueeze(model._forward_transform(model._stft(m.to(device))), 0)
        M = pad_spec(M)

        Noise = torch.unsqueeze(model._forward_transform(model._stft(noise.to(device))), 0)
        Noise = pad_spec(Noise)

        y = y * norm_factor
        x = x * norm_factor
        
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_start = time.perf_counter()

        with utterance_rng(args.seed, sample_id, device) as sampling_seed:
            with torch.no_grad():
                sample = model.reverse_sample(
                    Y, M, reverse_start_time=reverse_starting_point, noise_shape_ref=X
                )
        sample = sample.squeeze()
        x_hat = model.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_s = time.perf_counter() - t_start
        audio_len_s = float(min_leng) / float(sr)
        rtf = latency_s / max(audio_len_s, 1e-8)

        x_hat = x_hat.squeeze().detach().cpu().numpy()
        if args.correction_alpha != 1.0:
            x_hat = y + args.correction_alpha * (x_hat - y)
        if args.save_audio:
            files_dir = os.path.join(target_dir, "files")
            write(os.path.join(files_dir, filename), x_hat, 8000, subtype="FLOAT")
            shutil.copyfile(clean_file, os.path.join(files_dir, filename.split('_')[0] + '_ref.wav'))
            shutil.copyfile(mixture_file, os.path.join(files_dir, filename.split('_')[0] + '_mix.wav'))
            shutil.copyfile(noisy_file, os.path.join(files_dir, filename.split('_')[0] + '_pred.wav'))

        # Append metrics to data frame
        data["filename"].append(filename)
        try:
            p = pesq(sr, x, x_hat, 'nb')
        except: 
            p = float("nan")
        data["pesq"].append(p)
        data["estoi"].append(stoi(x, x_hat, sr, extended=True))
        data["si_sdr"].append(si_sdr(x, x_hat))
        data["latency_s"].append(latency_s)
        data["audio_len_s"].append(audio_len_s)
        data["rtf"].append(rtf)
        data["sampling_seed"].append(sampling_seed)
        if tibetan_refs:
            ref_info = tibetan_refs.get(sample_id, {"utt_id": "", "ref_text": ""})
            data.setdefault("utt_id", []).append(ref_info["utt_id"])
            data.setdefault("ref_text", []).append(ref_info["ref_text"])
            asr_manifest.append({
                "filename": filename,
                "sample_id": sample_id,
                "utt_id": ref_info["utt_id"],
                "ref_text": ref_info["ref_text"],
                "enhanced_wav": os.path.join(target_dir, "files", filename) if args.save_audio else "",
                "clean_wav": clean_file,
                "mix_wav": mixture_file,
                "sepformer_wav": noisy_file,
            })
        # wvmos = wvmos_model.calculate_one(target_dir + "files/" + filename)
        # data["WVMOS"].append(wvmos)


    # Save results as DataFrame
    df = pd.DataFrame(data)
    df["evaluation_seed"] = int(args.seed)
    df["sampling_policy"] = SAMPLING_POLICY
    df["correction_alpha"] = float(args.correction_alpha)
    df["reverse_starting_point"] = float(reverse_starting_point)
    metric_columns = ["pesq", "estoi", "si_sdr"]
    for column in metric_columns:
        df[column] = df[column].map(format_metric)
    df.to_csv(join(target_dir, "_results.csv"), index=False)
    if asr_manifest:
        pd.DataFrame(asr_manifest).to_csv(join(target_dir, "_asr_manifest.csv"), index=False, encoding="utf-8-sig")

    # Save average results
    text_file = join(target_dir, "_avg_results.txt")
    with open(text_file, 'w', encoding='utf-8') as file:
        file.write("PESQ: {} \n".format(mean_std_half_up(data["pesq"])))
        file.write("ESTOI: {} \n".format(mean_std_half_up(data["estoi"])))
        file.write("SI-SDR: {} \n".format(mean_std_half_up(data["si_sdr"])))
        file.write("Latency(s): {} \n".format(print_mean_std(data["latency_s"])))
        file.write("RTF: {} \n".format(print_mean_std(data["rtf"])))
        total_latency = float(np.sum(data["latency_s"])) if len(data["latency_s"]) > 0 else 0.0
        total_audio = float(np.sum(data["audio_len_s"])) if len(data["audio_len_s"]) > 0 else 0.0
        throughput = (len(data["latency_s"]) / total_latency) if total_latency > 0 else float("nan")
        global_rtf = (total_latency / total_audio) if total_audio > 0 else float("nan")
        file.write("Total latency(s): {:.4f} \n".format(total_latency))
        file.write("Total audio(s): {:.4f} \n".format(total_audio))
        file.write("Global RTF: {:.4f} \n".format(global_rtf))
        file.write("Throughput(files/s): {:.4f} \n".format(throughput))
        file.write("Correction alpha: {} \n".format(args.correction_alpha))
        if gmacs is not None:
            file.write("GMACs(model.dnn, per forward): {:.4f} \n".format(gmacs))
        # file.write("WVMOS: {} \n".format(print_mean_std(data["WVMOS"])))

    # Save settings
    text_file = join(target_dir, "_settings.txt")
    with open(text_file, 'w', encoding='utf-8') as file:
        file.write("checkpoint file: {}\n".format(checkpoint_file))
        file.write("Reverse steps: 1\n")
        file.write("Reverse starting point: {}\n".format(reverse_starting_point))
        file.write("Correction alpha: {}\n".format(args.correction_alpha))
        file.write("t_eps: {}\n".format(t_eps))
        file.write("seed: {}\n".format(args.seed))
        file.write("sampling policy: {}\n".format(SAMPLING_POLICY))
        file.write("save audio: {}\n".format(args.save_audio))
        file.write("tibetan label file: {}\n".format(args.tibetan_label_file))
        file.write("tibetan metadata file: {}\n".format(args.tibetan_metadata_file or os.path.join(args.test_dir, "metadata.csv")))
        file.write("device: {}\n".format(device))
        file.write("total params: {}\n".format(total_params))
        file.write("trainable params: {}\n".format(trainable_params))
        if gmacs is not None:
            file.write("GMACs(model.dnn, per forward): {:.4f}\n".format(gmacs))
        if device.type == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            file.write("peak cuda memory (MB): {:.2f}\n".format(peak_mem_mb))
