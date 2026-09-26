import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from function_script.syllable.diagnose.syllable_alignment import (
        compute_acoustic_boundary_score,
        map_boundaries_dp,
    )
except ImportError:  # pragma: no cover - supports direct execution from this directory
    from ..diagnose.syllable_alignment import compute_acoustic_boundary_score, map_boundaries_dp

try:
    import torch
    import torchaudio
except ImportError as e:
    raise SystemExit(
        "tibetan_syllable_mask.py requires torch and torchaudio. "
        "Run it inside the project environment from environment.yml/requirements.txt."
    ) from e


EPS = 1e-10
TIBETAN_PUNCT = " \t\r\n།༎༏༐༑༔༄༅༆༈༺༻༼༽.,!?;:()[]{}\"'“”‘’、，。！？；："


def parse_args():
    parser = argparse.ArgumentParser(
        description="E1 syllable-role mask generation and visualization for Tibetan Fast-GeCo."
    )
    parser.add_argument("--split_dir", type=str, default="zang_data/val")
    parser.add_argument("--label_file", type=str, default="zang_data/label.txt")
    parser.add_argument("--metadata_file", type=str, default="",
                        help="Defaults to <split_dir>/metadata.csv.")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Defaults to the role-mask directory below split_dir.")
    parser.add_argument("--sample_rate", type=int, default=8000)
    parser.add_argument("--n_fft", type=int, default=510)
    parser.add_argument("--hop_length", type=int, default=64)
    parser.add_argument("--max_files", type=int, default=0,
                        help="Use 0 for all rows.")
    parser.add_argument("--num_visualize", type=int, default=20)
    parser.add_argument("--highband_start_hz", type=float, default=1800.0)
    parser.add_argument("--vad_quantile", type=float, default=0.25)
    parser.add_argument("--transition_peak", type=float, default=0.6)
    parser.add_argument("--transition_floor", type=float, default=0.08)
    parser.add_argument("--nucleus_sigma_ratio", type=float, default=0.18)
    parser.add_argument("--boundary_refine_frames", type=int, default=5)
    parser.add_argument("--alignment_mode", choices=("legacy", "acoustic_dp"), default="legacy")
    parser.add_argument("--duration_prior_weight", type=float, default=0.2)
    parser.add_argument("--min_segment_ratio", type=float, default=0.25)
    parser.add_argument("--max_segment_ratio", type=float, default=4.0)
    args = parser.parse_args()
    return args


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def default_metadata_file(args):
    if args.metadata_file:
        return args.metadata_file
    return str(Path(args.split_dir) / "metadata.csv")


def default_output_dir(args):
    if args.output_dir:
        return args.output_dir
    directory = "role_masks_stage1_acoustic" if args.alignment_mode == "acoustic_dp" else "role_masks"
    return str(Path(args.split_dir) / directory)


def config_hash(config):
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_labels(label_file):
    labels = {}
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            utt_id, text = parts[0].strip(), parts[1].strip()
            if utt_id:
                labels[utt_id] = text
    return labels


def clean_syllable(syllable):
    cleaned = syllable.strip(TIBETAN_PUNCT)
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned


def split_tibetan_syllables(text):
    syllables = [clean_syllable(s) for s in str(text).split("་")]
    return [s for s in syllables if s]


def sample_id_from_metadata(value):
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def safe_output_id(value):
    text = sample_id_from_metadata(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def first_row_value(row, keys):
    for key in keys:
        value = row.get(key, "")
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return value
    return ""


def utt_id_from_speech_path(path):
    if not path or pd.isna(path):
        return ""
    text = str(path).strip().replace("\\", "/")
    parts = [p for p in text.split("/") if p]
    if len(parts) >= 2:
        parent = parts[-2]
        stem = Path(parts[-1]).stem
        return f"{parent}_{stem}"
    return Path(text).stem


def first_existing(paths):
    for path in paths:
        if not path:
            continue
        path = Path(str(path))
        if path.exists():
            return str(path)
    return ""


def clean_path_for_row(row, split_dir, sample_id):
    source1_path = first_row_value(row, ["source1_path", "clean_output", "clean_path"])
    source1_name = Path(str(source1_path)).name if source1_path else ""
    bucket_dir = sample_id_from_metadata(row.get("bucket_dir", ""))
    return first_existing([
        source1_path,
        f"/{source1_path}" if source1_path else "",
        Path(split_dir) / f"{sample_id}_source1.wav",
        Path(split_dir) / bucket_dir / f"{sample_id}_source1.wav" if bucket_dir else "",
        Path(split_dir) / bucket_dir / source1_name if bucket_dir and source1_name else "",
    ])


def load_audio(path, sample_rate):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)
    return wav.squeeze(0).float()


def normalize01(values):
    values = np.asarray(values, dtype=np.float64)
    out = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if out.size == 0:
        return out
    lo = float(np.min(out))
    hi = float(np.max(out))
    if hi - lo < EPS:
        return np.zeros_like(out)
    return (out - lo) / (hi - lo + EPS)


def match_length(values, target_len, fill=0.0):
    values = np.asarray(values)
    if values.size == target_len:
        return values
    dtype = values.dtype if values.size else np.float64
    out = np.full(target_len, fill, dtype=dtype)
    n = min(values.size, target_len)
    if n > 0:
        out[:n] = values[:n]
    return out


def stft_magnitude(wav, sample_rate, n_fft, hop_length):
    window = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        return_complex=True,
    )
    mag = spec.abs().transpose(0, 1).cpu().numpy()
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))
    return mag, freqs


def try_librosa_f0(wav_np, sample_rate, n_fft, hop_length):
    try:
        import librosa
    except Exception:
        return None, None, "fallback_proxy"

    try:
        f0, voiced_flag, _ = librosa.pyin(
            wav_np.astype(np.float32),
            fmin=50,
            fmax=min(500, sample_rate // 2 - 1),
            sr=sample_rate,
            frame_length=n_fft,
            hop_length=hop_length,
            center=True,
        )
        return np.nan_to_num(f0, nan=0.0).astype(np.float64), voiced_flag.astype(bool), "librosa.pyin"
    except Exception:
        return None, None, "fallback_proxy"


def compute_features(wav, args, estimate_f0=True):
    if wav.numel() < max(args.n_fft // 2, args.hop_length * 2):
        raise ValueError(f"audio_too_short: {wav.numel()} samples")

    mag, freqs = stft_magnitude(wav, args.sample_rate, args.n_fft, args.hop_length)
    num_frames = mag.shape[0]
    if num_frames < 2:
        raise ValueError(f"too_few_frames: {num_frames}")

    energy = np.mean(mag ** 2, axis=1)
    energy_norm = normalize01(energy)
    positive_energy = energy[energy > EPS]
    vad_thr = float(np.quantile(positive_energy, args.vad_quantile)) if positive_energy.size else 0.0
    vad = energy > max(vad_thr, EPS)

    flux = np.zeros(num_frames, dtype=np.float64)
    diff = np.diff(mag, axis=0)
    flux[1:] = np.mean(np.maximum(diff, 0.0), axis=1)
    flux_norm = normalize01(flux)

    high_mask = freqs >= args.highband_start_hz
    if not np.any(high_mask):
        high_mask = freqs >= (0.45 * (args.sample_rate / 2.0))
    mag_sum = np.sum(mag, axis=1) + EPS
    highband_ratio = np.sum(mag[:, high_mask], axis=1) / mag_sum
    highband_norm = normalize01(highband_ratio)

    peak_ratio = np.max(mag, axis=1) / (np.mean(mag + EPS, axis=1) + EPS)
    harmonicity = np.clip(np.log1p(peak_ratio) / math.log1p(100.0), 0.0, 1.0)

    if estimate_f0:
        wav_np = wav.detach().cpu().numpy().astype(np.float64)
        f0, voiced, f0_method = try_librosa_f0(
            wav_np, args.sample_rate, args.n_fft, args.hop_length
        )
    else:
        f0, voiced, f0_method = None, None, "harmonicity_proxy"
    if f0 is None:
        harmonic_thr = quantile(harmonicity[vad], 0.60, default=0.5)
        voiced = vad & (harmonicity >= harmonic_thr)
        f0 = np.zeros(num_frames, dtype=np.float64)
    else:
        f0 = match_length(f0, num_frames, fill=0.0).astype(np.float64)
        voiced = match_length(voiced, num_frames, fill=False).astype(bool)

    return {
        "num_frames": num_frames,
        "spectral_magnitude": mag,
        "energy": energy,
        "energy_norm": energy_norm,
        "vad_threshold": vad_thr,
        "vad": vad.astype(bool),
        "flux": flux,
        "flux_norm": flux_norm,
        "highband_ratio": highband_ratio,
        "highband_norm": highband_norm,
        "harmonicity": harmonicity,
        "f0": f0,
        "voiced": voiced.astype(bool),
        "f0_method": f0_method,
    }


def quantile(values, q, default=0.0):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return default
    return float(np.quantile(values, q))


def detect_active_bounds(features, minimum_active_frames=3):
    """Historical full-mask bounds using VAD, voicing, and relative energy."""
    num_frames = features["num_frames"]
    energy_norm = features["energy_norm"]
    vad = features["vad"]
    voiced = features["voiced"]
    active = vad | voiced | (energy_norm >= quantile(energy_norm, 0.60, default=0.0))

    active_idx = np.flatnonzero(active)
    min_active = max(1, min(num_frames, int(minimum_active_frames)))
    if active_idx.size < min_active:
        return 0, num_frames - 1, False
    return int(active_idx[0]), int(active_idx[-1]), True


def active_bounds(features, syllable_count):
    """Compatibility wrapper for the existing full-mask implementation."""
    return detect_active_bounds(features, max(3, int(syllable_count)))


def refine_boundaries(bounds, energy_norm, start, end, refine_frames):
    bounds = np.asarray(bounds, dtype=int).copy()
    refine = max(0, int(refine_frames))
    if refine <= 0 or bounds.size <= 2:
        return bounds

    for i in range(1, bounds.size - 1):
        low = max(start + 1, bounds[i] - refine, bounds[i - 1] + 1)
        high = min(end, bounds[i] + refine, bounds[i + 1] - 1)
        if high < low:
            continue
        candidates = np.arange(low, high + 1)
        local_energy = energy_norm[candidates]
        bounds[i] = int(candidates[int(np.argmin(local_energy))])
    return bounds


def make_span_boundaries(start, end, syllable_count, energy_norm=None, refine_frames=0):
    usable = max(1, end - start + 1)
    count = max(1, int(syllable_count))
    raw = np.linspace(start, end + 1, count + 1)
    bounds = np.rint(raw).astype(int)
    bounds[0] = start
    bounds[-1] = end + 1
    if energy_norm is not None:
        bounds = refine_boundaries(bounds, energy_norm, start, end, refine_frames)
    for i in range(1, len(bounds)):
        if bounds[i] <= bounds[i - 1]:
            bounds[i] = bounds[i - 1] + 1
    if bounds[-1] > end + 1:
        bounds = np.linspace(start, end + 1, count + 1).astype(int)
        bounds[0] = start
        bounds[-1] = end + 1
    spans = []
    for i in range(count):
        s = int(np.clip(bounds[i], 0, end))
        e = int(np.clip(bounds[i + 1], s + 1, end + 1))
        spans.append((s, e))
    if usable < count:
        spans = [(min(start + i, end), min(start + i + 1, end + 1)) for i in range(count)]
    return spans


def gaussian_window(indices, center, sigma):
    sigma = max(float(sigma), 1.0)
    return np.exp(-0.5 * ((indices - float(center)) / sigma) ** 2)


def transition_edge_weights(size, peak, floor, reverse=False):
    if size <= 0:
        return np.zeros(0, dtype=np.float64)
    if size == 1:
        values = np.array([peak], dtype=np.float64)
    else:
        values = np.linspace(peak, floor, size, dtype=np.float64)
    if reverse:
        values = values[::-1]
    return values


def _mask_arrays(num_frames):
    return tuple(np.zeros(num_frames, dtype=np.float64) for _ in range(4))


def _finish_masks(onset, nucleus, transition, valid):
    return {
        "onset_like": np.clip(onset * valid, 0.0, 1.0).astype(np.float32),
        "nucleus_like": np.clip(nucleus * valid, 0.0, 1.0).astype(np.float32),
        "transition_like": np.clip(transition * valid, 0.0, 1.0).astype(np.float32),
        "valid_mask": np.clip(valid, 0.0, 1.0).astype(np.float32),
    }


def _render_transition(transition, idx, args):
    length = idx.size
    transition_len = max(1, int(math.ceil(0.12 * length)))
    peak = float(np.clip(args.transition_peak, 0.0, 1.0))
    floor = float(np.clip(args.transition_floor, 0.0, peak))
    left_idx = idx[:transition_len]
    right_idx = idx[-transition_len:]
    transition[left_idx] = np.maximum(
        transition[left_idx],
        transition_edge_weights(left_idx.size, peak, floor, reverse=False),
    )
    transition[right_idx] = np.maximum(
        transition[right_idx],
        transition_edge_weights(right_idx.size, peak, floor, reverse=True),
    )


def render_acoustic_masks(spans, features, args):
    """Render acoustic role shapes within transcript-guided spans."""
    num_frames = features["num_frames"]
    onset, nucleus, transition, valid = _mask_arrays(num_frames)
    energy_norm = features["energy_norm"]
    flux_norm = features["flux_norm"]
    highband_norm = features["highband_norm"]
    harmonicity = features["harmonicity"]
    voiced = features["voiced"].astype(np.float64)
    nucleus_sigma_ratio = max(0.05, float(args.nucleus_sigma_ratio))
    for s, e in spans:
        idx = np.arange(s, e)
        if idx.size == 0:
            continue
        length = idx.size
        valid[idx] = 1.0

        local_score = energy_norm[idx] + 0.35 * voiced[idx] + 0.25 * harmonicity[idx]
        center = int(idx[int(np.argmax(local_score))])

        onset_len = max(1, int(math.ceil(0.25 * length)))
        nucleus_sigma = max(1.0, nucleus_sigma_ratio * length)

        onset_idx = idx[:onset_len]
        onset_weight = 0.45 + 0.35 * flux_norm[onset_idx] + 0.20 * highband_norm[onset_idx]
        onset[onset_idx] = np.maximum(onset[onset_idx], onset_weight)

        nucleus_weight = gaussian_window(idx, center, nucleus_sigma)
        nucleus_gate = 0.25 + 0.35 * energy_norm[idx] + 0.25 * voiced[idx] + 0.15 * harmonicity[idx]
        nucleus_weight *= nucleus_gate
        nucleus[idx] = np.maximum(nucleus[idx], nucleus_weight)
        _render_transition(transition, idx, args)
    return _finish_masks(onset, nucleus, transition, valid)


def generate_role_masks(features, syllable_count, args):
    """Generate transcript-guided role masks."""
    start, end, used_vad_bounds = active_bounds(features, syllable_count)
    if args.alignment_mode == "acoustic_dp":
        alignment = map_boundaries_dp(
            compute_acoustic_boundary_score(features), syllable_count, start, end + 1,
            min_segment_ratio=args.min_segment_ratio,
            max_segment_ratio=args.max_segment_ratio,
            duration_prior_weight=args.duration_prior_weight,
        )
        if not alignment.success:
            raise ValueError(f"acoustic_alignment_failed: {alignment.reason}")
        spans = list(alignment.spans)
        boundaries = alignment.boundaries
    else:
        spans = make_span_boundaries(
            start, end, syllable_count,
            energy_norm=features["energy_norm"],
            refine_frames=args.boundary_refine_frames,
        )
        boundaries = tuple(int(span[1]) for span in spans[:-1])
    masks = render_acoustic_masks(spans, features, args)
    masks.update({
        "used_vad_bounds": used_vad_bounds,
        "active_start_frame": start,
        "active_end_frame": end,
        "boundary_frames": boundaries,
        "nucleus_peak_frames": tuple(),
    })
    return masks


def masked_ratio(mask):
    mask = np.asarray(mask, dtype=np.float64)
    if mask.size == 0:
        return float("nan")
    return float(np.mean(mask > 0.05))


def mask_mean(mask):
    mask = np.asarray(mask, dtype=np.float64)
    if mask.size == 0:
        return float("nan")
    return float(np.mean(mask))


def save_npz(path, masks, record):
    np.savez_compressed(
        path,
        onset_like=masks["onset_like"],
        nucleus_like=masks["nucleus_like"],
        transition_like=masks["transition_like"],
        valid_mask=masks["valid_mask"],
        utt_id=np.array(record["utt_id"]),
        sample_id=np.array(record["sample_id"]),
        syllable_count=np.array(record["syllable_count"], dtype=np.int32),
        syllable_count_source=np.array(record["syllable_count_source"]),
        mask_mode=np.array(record["mask_mode"]),
        config_hash=np.array(record["config_hash"]),
        num_frames=np.array(record["num_frames"], dtype=np.int32),
        hop_length=np.array(record["hop_length"], dtype=np.int32),
        n_fft=np.array(record["n_fft"], dtype=np.int32),
        sample_rate=np.array(record["sample_rate"], dtype=np.int32),
        f0_method=np.array(record["f0_method"]),
        transition_peak=np.array(record["transition_peak"], dtype=np.float32),
        transition_floor=np.array(record["transition_floor"], dtype=np.float32),
        nucleus_sigma_ratio=np.array(record["nucleus_sigma_ratio"], dtype=np.float32),
        boundary_refine_frames=np.array(record["boundary_refine_frames"], dtype=np.int32),
        used_vad_bounds=np.array(masks["used_vad_bounds"], dtype=np.bool_),
        active_start_frame=np.array(masks["active_start_frame"], dtype=np.int32),
        active_end_frame=np.array(masks["active_end_frame"], dtype=np.int32),
        alignment_mode=np.array(record["alignment_mode"]),
        nucleus_peak_frames=np.asarray(masks["nucleus_peak_frames"], dtype=np.int32),
        boundary_frames=np.asarray(masks["boundary_frames"], dtype=np.int32),
        duration_prior_weight=np.array(record["duration_prior_weight"], dtype=np.float32),
        min_segment_ratio=np.array(record["min_segment_ratio"], dtype=np.float32),
        max_segment_ratio=np.array(record["max_segment_ratio"], dtype=np.float32),
    )


def visualize(path, wav, features, masks, record, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_frames = features["num_frames"]
    frame_t = np.arange(num_frames) * float(args.hop_length) / float(args.sample_rate)
    wav_np = wav.detach().cpu().numpy()
    wav_t = np.arange(wav_np.size) / float(args.sample_rate)

    energy = normalize01(features["energy"])
    flux = normalize01(features["flux"])
    f0 = features["f0"]
    f0_plot = f0.copy()
    f0_plot[f0_plot <= 0] = np.nan

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        f"sample={record['sample_id']} utt={record['utt_id']} "
        f"syllables={record['syllable_count']} f0={record['f0_method']}"
    )
    axes[0].plot(wav_t, wav_np, color="black", linewidth=0.7)
    axes[0].set_ylabel("wav")
    axes[0].grid(alpha=0.2)

    axes[1].plot(frame_t, energy, label="energy", color="#1f77b4")
    axes[1].plot(frame_t, flux, label="flux", color="#d62728", alpha=0.8)
    axes[1].fill_between(frame_t, 0, features["vad"].astype(float), color="#2ca02c", alpha=0.15, label="vad")
    axes[1].set_ylabel("features")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.2)

    axes[2].plot(frame_t, f0_plot, color="#9467bd", linewidth=0.9)
    axes[2].fill_between(frame_t, 0, features["voiced"].astype(float) * np.nanmax(f0_plot)
                         if np.any(np.isfinite(f0_plot)) else features["voiced"].astype(float),
                         color="#9467bd", alpha=0.12, label="voiced")
    axes[2].set_ylabel("F0 Hz")
    axes[2].grid(alpha=0.2)

    axes[3].plot(frame_t, masks["onset_like"], label="onset_like", color="#d62728")
    axes[3].plot(frame_t, masks["nucleus_like"], label="nucleus_like", color="#2ca02c")
    axes[3].plot(frame_t, masks["transition_like"], label="transition_like", color="#ff7f0e")
    axes[3].plot(frame_t, masks["valid_mask"], label="valid_mask", color="#7f7f7f", alpha=0.4)
    axes[3].set_ylabel("mask")
    axes[3].set_xlabel("time (s)")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].legend(loc="upper right")
    axes[3].grid(alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(path, summary_df, skipped_df, args):
    lines = []
    lines.append("# E1 Syllable-Role Mask Report")
    lines.append("")
    lines.append(f"- split_dir: `{args.split_dir}`")
    lines.append("- mask_mode: `full`")
    lines.append(f"- sample_rate: `{args.sample_rate}`")
    lines.append(f"- n_fft: `{args.n_fft}`")
    lines.append(f"- hop_length: `{args.hop_length}`")
    lines.append(f"- transition_peak: `{args.transition_peak}`")
    lines.append(f"- transition_floor: `{args.transition_floor}`")
    lines.append(f"- nucleus_sigma_ratio: `{args.nucleus_sigma_ratio}`")
    lines.append(f"- boundary_refine_frames: `{args.boundary_refine_frames}`")
    lines.append(f"- alignment_mode: `{args.alignment_mode}`")
    lines.append(f"- duration_prior_weight: `{args.duration_prior_weight}`")
    lines.append(f"- min_segment_ratio: `{args.min_segment_ratio}`")
    lines.append(f"- max_segment_ratio: `{args.max_segment_ratio}`")
    lines.append("")
    total = len(summary_df)
    success = int((summary_df["status"] == "ok").sum()) if total else 0
    skipped = total - success
    lines.append(f"- total_rows: `{total}`")
    lines.append(f"- success: `{success}`")
    lines.append(f"- skipped: `{skipped}`")
    lines.append("")

    ok = summary_df[summary_df["status"] == "ok"].copy() if total else pd.DataFrame()
    if len(ok) > 0:
        lines.append("## Successful Mask Statistics")
        for col in (
            "syllable_count", "num_frames", "vad_ratio",
            "onset_cover_ratio", "nucleus_cover_ratio", "transition_cover_ratio",
            "onset_mean", "nucleus_mean", "transition_mean",
        ):
            values = ok[col].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size:
                lines.append(f"- {col}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")
        lines.append("")
        lines.append(f"- f0_method_counts: `{ok['f0_method'].value_counts(dropna=False).to_dict()}`")
        lines.append(f"- used_vad_bounds_counts: `{ok['used_vad_bounds'].value_counts(dropna=False).to_dict()}`")
        lines.append("")

    if len(skipped_df) > 0:
        lines.append("## Skipped / Failed Samples")
        reason_counts = skipped_df["reason"].value_counts(dropna=False).to_dict()
        lines.append(f"- reason_counts: `{reason_counts}`")
        lines.append("")
        preview = skipped_df[["sample_id", "utt_id", "reason"]].head(20)
        for _, row in preview.iterrows():
            lines.append(f"- sample_id={row.get('sample_id', '')}, utt_id={row.get('utt_id', '')}, reason={row.get('reason', '')}")
        lines.append("")

    lines.append("## Interpretation")
    lines.append("- This is an offline E1 diagnostic artifact, not a training change.")
    lines.append("- If visualizations show systematic offsets, fix mask alignment before entering E2.")
    lines.append("- If a sample lacks a valid mask in later training, it should fall back to the original Fast-GeCo loss.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def process_row(
    row, labels, args, output_dir, viz_dir, visualize_sample,
    mask_config_hash="",
):
    sample_id = sample_id_from_metadata(first_row_value(row, ["id", "sample_id"]))
    bucket_dir = sample_id_from_metadata(row.get("bucket_dir", ""))
    output_id = safe_output_id(f"{bucket_dir}_{sample_id}" if bucket_dir else sample_id)
    speech_path = first_row_value(
        row,
        ["speech_path", "clean_input", "clean_path", "source_path", "wav_path", "path"],
    )
    utt_id = utt_id_from_speech_path(speech_path)
    base = {
        "sample_id": sample_id,
        "utt_id": utt_id,
        "status": "skipped",
        "reason": "",
        "clean_path": "",
        "text": "",
        "syllable_count": 0,
        "syllable_count_source": "",
        "mask_mode": 'full',
        "config_hash": mask_config_hash,
        "num_frames": 0,
        "vad_ratio": float("nan"),
        "onset_cover_ratio": float("nan"),
        "nucleus_cover_ratio": float("nan"),
        "transition_cover_ratio": float("nan"),
        "onset_mean": float("nan"),
        "nucleus_mean": float("nan"),
        "transition_mean": float("nan"),
        "f0_method": "",
        "used_vad_bounds": False,
        "active_start_frame": -1,
        "active_end_frame": -1,
        "alignment_mode": args.alignment_mode,
        "nucleus_peak_frames": np.zeros(0, dtype=np.int32),
        "boundary_frames": np.zeros(0, dtype=np.int32),
        "duration_prior_weight": args.duration_prior_weight,
        "min_segment_ratio": args.min_segment_ratio,
        "max_segment_ratio": args.max_segment_ratio,
        "mask_path": "",
        "visualization_path": "",
    }

    if not sample_id:
        base["reason"] = "missing_sample_id"
        return base
    if not utt_id:
        base["reason"] = "missing_utt_id"
        return base
    if utt_id not in labels:
        base["reason"] = "missing_label"
        return base
    text = labels[utt_id]
    syllables = split_tibetan_syllables(text)
    base["text"] = text
    syllable_count = len(syllables)
    syllable_count_source = "transcript_count"
    base["syllable_count"] = syllable_count
    if not syllables:
        base["reason"] = "no_valid_syllables"
        return base

    clean_path = clean_path_for_row(row, args.split_dir, sample_id)
    base["clean_path"] = clean_path
    if not clean_path:
        base["reason"] = "missing_clean_wav"
        return base

    try:
        wav = load_audio(clean_path, args.sample_rate)
        features = compute_features(wav, args)
        masks = generate_role_masks(features, syllable_count, args)
    except Exception as e:
        base["reason"] = f"feature_error: {e}"
        return base

    record = {
        "utt_id": utt_id,
        "sample_id": sample_id,
        "syllable_count": syllable_count,
        "syllable_count_source": syllable_count_source,
        "mask_mode": 'full',
        "config_hash": mask_config_hash,
        "num_frames": features["num_frames"],
        "hop_length": args.hop_length,
        "n_fft": args.n_fft,
        "sample_rate": args.sample_rate,
        "f0_method": features["f0_method"],
        "transition_peak": args.transition_peak,
        "transition_floor": args.transition_floor,
        "nucleus_sigma_ratio": args.nucleus_sigma_ratio,
        "boundary_refine_frames": args.boundary_refine_frames,
        "alignment_mode": args.alignment_mode,
        "duration_prior_weight": args.duration_prior_weight,
        "min_segment_ratio": args.min_segment_ratio,
        "max_segment_ratio": args.max_segment_ratio,
    }
    mask_path = Path(output_dir) / f"{output_id}_role_mask.npz"
    save_npz(mask_path, masks, record)

    visualization_path = ""
    if visualize_sample:
        visualization_path = str(Path(viz_dir) / f"{output_id}_role_mask.png")
        visualize(visualization_path, wav, features, masks, record, args)

    base.update({
        "status": "ok",
        "reason": "",
        "num_frames": features["num_frames"],
        "vad_ratio": float(np.mean(features["vad"])),
        "onset_cover_ratio": masked_ratio(masks["onset_like"]),
        "nucleus_cover_ratio": masked_ratio(masks["nucleus_like"]),
        "transition_cover_ratio": masked_ratio(masks["transition_like"]),
        "onset_mean": mask_mean(masks["onset_like"]),
        "nucleus_mean": mask_mean(masks["nucleus_like"]),
        "transition_mean": mask_mean(masks["transition_like"]),
        "f0_method": features["f0_method"],
        "used_vad_bounds": bool(masks["used_vad_bounds"]),
        "active_start_frame": int(masks["active_start_frame"]),
        "active_end_frame": int(masks["active_end_frame"]),
        "boundary_frames": list(masks["boundary_frames"]),
        "nucleus_peak_frames": list(masks["nucleus_peak_frames"]),
        "syllable_count_source": syllable_count_source,
        "mask_path": str(mask_path),
        "visualization_path": visualization_path,
    })
    return base


def effective_mask_config(args):
    config = {
        "mask_mode": 'full',
        "sample_rate": args.sample_rate,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "vad_quantile": args.vad_quantile,
        "transition_peak": args.transition_peak,
        "transition_floor": args.transition_floor,
        "nucleus_sigma_ratio": args.nucleus_sigma_ratio,
    }
    config.update({
        "alignment_mode": args.alignment_mode,
        "boundary_refine_frames": args.boundary_refine_frames,
        "duration_prior_weight": args.duration_prior_weight,
        "min_segment_ratio": args.min_segment_ratio,
        "max_segment_ratio": args.max_segment_ratio,
    })
    return config


def main():
    args = parse_args()
    if args.duration_prior_weight < 0:
        raise ValueError("duration_prior_weight must be non-negative")
    if args.min_segment_ratio <= 0 or args.max_segment_ratio < args.min_segment_ratio:
        raise ValueError("segment ratios must satisfy 0 < min_segment_ratio <= max_segment_ratio")
    frozen_config = effective_mask_config(args)
    frozen_config_hash = config_hash(frozen_config)
    metadata_file = default_metadata_file(args)
    output_dir = default_output_dir(args)
    viz_dir = Path(output_dir) / "visualizations"
    ensure_dir(output_dir)
    ensure_dir(viz_dir)

    labels = read_labels(args.label_file)
    metadata = pd.read_csv(metadata_file)
    if args.max_files and args.max_files > 0:
        metadata = metadata.head(args.max_files)

    rows = []
    visualize_remaining = max(0, args.num_visualize)
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="E1 syllable masks"):
        visualize_sample = visualize_remaining > 0
        result = process_row(
            row, labels, args, output_dir, viz_dir, visualize_sample,
            mask_config_hash=frozen_config_hash,
        )
        if result["status"] == "ok" and visualize_sample:
            visualize_remaining -= 1
        rows.append(result)

    summary_df = pd.DataFrame(rows)
    skipped_df = summary_df[summary_df["status"] != "ok"].copy()

    summary_path = Path(output_dir) / "mask_summary.csv"
    skipped_path = Path(output_dir) / "skipped_samples.csv"
    report_path = Path(output_dir) / "mask_report.md"
    config_path = Path(output_dir) / "mask_config.json"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    skipped_df.to_csv(skipped_path, index=False, encoding="utf-8-sig")
    write_report(report_path, summary_df, skipped_df, args)
    config_path.write_text(
        json.dumps(
            {"config_hash": frozen_config_hash, "config": frozen_config},
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    ok_count = int((summary_df["status"] == "ok").sum()) if len(summary_df) else 0
    print(f"Wrote {summary_path}")
    print(f"Wrote {skipped_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {config_path}")
    print(f"Generated {ok_count}/{len(summary_df)} masks in {output_dir}")


if __name__ == "__main__":
    main()
