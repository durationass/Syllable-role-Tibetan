import argparse
import os
import shutil
from pathlib import Path

import torch
import torchaudio
from speechbrain.inference.separation import SepformerSeparation
from tqdm import tqdm


EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PIT-selected source1hatP with fine-tuned SepFormer.")
    parser.add_argument("--input_root", type=str, default="zang_data")
    parser.add_argument("--output_root", type=str, default="zang_data_sepformer_ft")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Optional checkpoint from finetune_sepformer_tibetan.py. If omitted, use the original SepFormer from_hparams() model.")
    parser.add_argument("--source", type=str, default="pretrained_models/sepformer-whamr")
    parser.add_argument("--savedir", type=str, default="pretrained_models/sepformer-whamr")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sample_rate", type=int, default=8000)
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def si_sdr(estimate, reference):
    estimate = estimate.reshape(-1)
    reference = reference.reshape(-1)
    min_len = min(estimate.numel(), reference.numel())
    estimate = estimate[:min_len] - estimate[:min_len].mean()
    reference = reference[:min_len] - reference[:min_len].mean()
    projection = torch.dot(estimate, reference) * reference / (torch.dot(reference, reference) + EPS)
    noise = estimate - projection
    return 10.0 * torch.log10((projection.pow(2).sum() + EPS) / (noise.pow(2).sum() + EPS))


def load_reference(path, sample_rate):
    wav, sr = torchaudio.load(str(path))
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)
    return wav.cpu()


def separate_mix_file(model, path, sample_rate, device):
    mix, sr = torchaudio.load(str(path))
    if mix.size(0) > 1:
        mix = mix.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        mix = torchaudio.transforms.Resample(sr, sample_rate)(mix)
    mix = mix.to(device)
    est_sources = model.separate_batch(mix)
    est_sources = est_sources / est_sources.abs().max(dim=1, keepdim=True)[0].clamp_min(EPS)
    return est_sources


def select_speech_channel(est_sources, reference):
    best_idx = 0
    best_score = None
    for idx in range(est_sources.shape[-1]):
        candidate = est_sources[:, :, idx].detach().cpu()
        score = si_sdr(candidate, reference)
        if best_score is None or score > best_score:
            best_score = score
            best_idx = idx
    return est_sources[:, :, best_idx].detach().cpu(), best_idx, best_score


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.mods.encoder.load_state_dict(ckpt["encoder"])
    model.mods.masknet.load_state_dict(ckpt["masknet"])
    model.mods.decoder.load_state_dict(ckpt["decoder"])
    model.mods.eval()


def ensure_clean_output_dir(path, overwrite):
    path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for file_path in path.glob("*_source1hatP.wav"):
            file_path.unlink()


def copy_pair(input_split, output_split, mix_file):
    sample_id = mix_file.name.replace("_mix.wav", "")
    src_mix = mix_file
    src_clean = input_split / f"{sample_id}_source1.wav"
    dst_mix = output_split / src_mix.name
    dst_clean = output_split / src_clean.name
    if not src_clean.exists():
        raise FileNotFoundError(f"Missing clean source: {src_clean}")
    if not dst_mix.exists():
        shutil.copyfile(src_mix, dst_mix)
    if not dst_clean.exists():
        shutil.copyfile(src_clean, dst_clean)
    return sample_id, dst_mix, dst_clean


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model = SepformerSeparation.from_hparams(
        source=args.source,
        savedir=args.savedir,
        run_opts={"device": str(device)},
    )
    if args.checkpoint.strip():
        load_checkpoint(model, args.checkpoint, device)
    else:
        model.mods.eval()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]

    for split in splits:
        input_split = input_root / split
        output_split = output_root / split
        if not input_split.is_dir():
            print(f"Skip missing split: {input_split}")
            continue
        ensure_clean_output_dir(output_split, args.overwrite)
        mix_files = sorted(input_split.glob("*_mix.wav"))
        if args.max_files and args.max_files > 0:
            mix_files = mix_files[:args.max_files]
        selected_counts = {}
        print(f"\nProcessing {split}: {len(mix_files)} files")
        for mix_file in tqdm(mix_files):
            sample_id, dst_mix, dst_clean = copy_pair(input_split, output_split, mix_file)
            hatp_path = output_split / f"{sample_id}_source1hatP.wav"
            if hatp_path.exists() and not args.overwrite:
                continue
            reference = load_reference(dst_clean, args.sample_rate)
            with torch.no_grad():
                est_sources = separate_mix_file(model, dst_mix, args.sample_rate, device)
            source1_hat, selected_idx, _ = select_speech_channel(est_sources, reference)
            selected_counts[selected_idx] = selected_counts.get(selected_idx, 0) + 1
            torchaudio.save(str(hatp_path), source1_hat, args.sample_rate)
        print(f"Selected channel counts for {split}: {selected_counts}")

    print(f"\nDone. New front-end data saved to: {output_root}")


if __name__ == "__main__":
    main()
