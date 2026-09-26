import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn


REQUIRED_MASK_ARRAYS = (
    "onset_like",
    "nucleus_like",
    "transition_like",
    "valid_mask",
)
VERSIONED_MASK_FIELDS = ("mask_mode", "alignment_mode", "config_hash")
LEGACY_FINGERPRINT_FIELDS = (
    "transition_peak",
    "transition_floor",
    "nucleus_sigma_ratio",
    "boundary_refine_frames",
)


def _npz_scalar(data, key):
    if key not in data.files:
        raise ValueError(f"missing metadata field {key!r}")
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"metadata field {key!r} must be scalar")
    value = value.reshape(()).item()
    return value.decode("utf-8") if isinstance(value, bytes) else value


def validate_role_mask_directory(
    role_mask_dir,
    sample_ids,
    sample_rate=8000,
    n_fft=510,
    hop_length=64,
    mask_mode="full",
    alignment_mode="legacy",
    minimum_coverage=0.95,
):
    """Validate one frozen mask set before starting role-aware training."""
    mask_dir = os.path.abspath(os.fspath(role_mask_dir))
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Role mask directory does not exist: {mask_dir}")

    expected_ids = {str(sample_id) for sample_id in sample_ids}
    mask_paths = sorted(
        os.path.join(mask_dir, name)
        for name in os.listdir(mask_dir)
        if name.endswith("_role_mask.npz")
    )
    if not mask_paths:
        raise RuntimeError(f"No *_role_mask.npz files found under {mask_dir}")

    found_ids = set()
    schemas = set()
    config_hashes = set()
    legacy_configs = set()
    for path in mask_paths:
        try:
            with np.load(path, allow_pickle=False) as data:
                arrays = []
                for key in REQUIRED_MASK_ARRAYS:
                    if key not in data.files:
                        raise ValueError(f"missing array {key!r}")
                    values = np.asarray(data[key])
                    if values.ndim != 1 or values.size == 0:
                        raise ValueError(f"array {key!r} must be a non-empty vector")
                    arrays.append(values)
                lengths = {values.size for values in arrays}
                if len(lengths) != 1:
                    raise ValueError("role mask arrays must have identical lengths")
                array_length = next(iter(lengths))

                actual = {
                    "sample_rate": int(_npz_scalar(data, "sample_rate")),
                    "n_fft": int(_npz_scalar(data, "n_fft")),
                    "hop_length": int(_npz_scalar(data, "hop_length")),
                }
                expected = {
                    "sample_rate": int(sample_rate),
                    "n_fft": int(n_fft),
                    "hop_length": int(hop_length),
                }
                if actual != expected:
                    raise ValueError(
                        f"mask metadata mismatch: expected {expected}, got {actual}"
                    )
                num_frames = int(_npz_scalar(data, "num_frames"))
                if num_frames != array_length:
                    raise ValueError(
                        f"num_frames={num_frames} does not match array length {array_length}"
                    )
                sample_id = str(_npz_scalar(data, "sample_id"))
                filename_id = os.path.basename(path)[: -len("_role_mask.npz")]
                if sample_id != filename_id:
                    raise ValueError(
                        f"sample_id {sample_id!r} does not match filename ID {filename_id!r}"
                    )

                version_fields_present = tuple(
                    key in data.files for key in VERSIONED_MASK_FIELDS
                )
                if all(version_fields_present):
                    schema = "versioned_v2"
                    versioned_actual = {
                        "mask_mode": str(_npz_scalar(data, "mask_mode")),
                        "alignment_mode": str(_npz_scalar(data, "alignment_mode")),
                    }
                    versioned_expected = {
                        "mask_mode": str(mask_mode),
                        "alignment_mode": str(alignment_mode),
                    }
                    if versioned_actual != versioned_expected:
                        raise ValueError(
                            "mask version metadata mismatch: "
                            f"expected {versioned_expected}, got {versioned_actual}"
                        )
                    config_hashes.add(str(_npz_scalar(data, "config_hash")))
                elif not any(version_fields_present):
                    schema = "legacy_v1"
                    if str(mask_mode) != "full" or str(alignment_mode) != "legacy":
                        raise ValueError(
                            "Unversioned legacy masks can only be validated as "
                            "mask_mode='full', alignment_mode='legacy'"
                        )
                    legacy_config = tuple(
                        float(_npz_scalar(data, key))
                        if key != "boundary_refine_frames"
                        else int(_npz_scalar(data, key))
                        for key in LEGACY_FINGERPRINT_FIELDS
                    )
                    legacy_configs.add(legacy_config)
                else:
                    present = [
                        key
                        for key, is_present in zip(
                            VERSIONED_MASK_FIELDS, version_fields_present
                        )
                        if is_present
                    ]
                    raise ValueError(
                        "partially versioned mask metadata; "
                        f"present fields: {present}"
                    )
                schemas.add(schema)
        except Exception as exc:
            raise ValueError(f"Invalid role mask {path}: {exc}") from exc
        found_ids.add(sample_id)

    if len(schemas) != 1:
        raise ValueError(f"Role mask directory mixes schemas: {sorted(schemas)}")
    schema = next(iter(schemas))
    if schema == "versioned_v2":
        if len(config_hashes) != 1:
            raise ValueError(
                f"Role mask directory mixes config hashes: {sorted(config_hashes)}"
            )
        config_hash = next(iter(config_hashes))
    else:
        if len(legacy_configs) != 1:
            raise ValueError(
                "Role mask directory mixes legacy construction parameters: "
                f"{sorted(legacy_configs)}"
            )
        legacy_config = next(iter(legacy_configs))
        fingerprint_payload = {
            "schema": schema,
            "sample_rate": int(sample_rate),
            "n_fft": int(n_fft),
            "hop_length": int(hop_length),
            **dict(zip(LEGACY_FINGERPRINT_FIELDS, legacy_config)),
        }
        digest = hashlib.sha256(
            json.dumps(
                fingerprint_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        config_hash = f"legacy_v1:{digest}"
    matched = expected_ids & found_ids
    coverage = len(matched) / max(len(expected_ids), 1)
    missing_ids = sorted(expected_ids - found_ids)
    if coverage < float(minimum_coverage):
        raise ValueError(
            f"Role mask coverage {coverage:.2%} is below {minimum_coverage:.2%}; "
            f"missing examples: {missing_ids[:10]}"
        )
    return {
        "mask_dir": mask_dir,
        "schema": schema,
        "config_hash": config_hash,
        "coverage": coverage,
        "matched": len(matched),
        "missing": len(missing_ids),
        "missing_ids": missing_ids,
    }


class SyllableRoleLoss(nn.Module):
    def __init__(
        self,
        role_mask_dir,
        sample_rate=8000,
        n_fft=510,
        hop_length=64,
        highband_start_hz=1800.0,
        nucleus_low_hz=300.0,
        nucleus_high_hz=3400.0,
        onset_weight=3.0,
        nucleus_weight=2.0,
        transition_weight=0.0,
        onset_low_hz=0.0,
        onset_high_hz=None,
        frequency_nucleus_low_hz=0.0,
        frequency_nucleus_high_hz=None,
        crop_alignment="legacy_floor",
        strict_loading=False,
        eps=1e-8,
    ):
        super().__init__()
        self.role_mask_dir = role_mask_dir
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.highband_start_hz = highband_start_hz
        self.nucleus_low_hz = nucleus_low_hz
        self.nucleus_high_hz = nucleus_high_hz
        self.onset_weight = onset_weight
        self.nucleus_weight = nucleus_weight
        self.transition_weight = transition_weight
        self.onset_low_hz = float(onset_low_hz)
        self.onset_high_hz = float(sample_rate / 2.0 if onset_high_hz is None else onset_high_hz)
        self.frequency_nucleus_low_hz = float(frequency_nucleus_low_hz)
        self.frequency_nucleus_high_hz = float(
            sample_rate / 2.0
            if frequency_nucleus_high_hz is None
            else frequency_nucleus_high_hz
        )
        # Alias retained for callers that use the internal role-weight helper.
        self.role_nucleus_low_hz = self.frequency_nucleus_low_hz
        self.role_nucleus_high_hz = self.frequency_nucleus_high_hz
        if crop_alignment not in ("legacy_floor", "fractional_linear"):
            raise ValueError(
                "crop_alignment must be 'legacy_floor' or 'fractional_linear'"
            )
        self.crop_alignment = crop_alignment
        self.strict_loading = bool(strict_loading)
        self.eps = eps
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self._freqs = None
        self._mask_cache = {}

    def _mask_path(self, sample_id):
        return os.path.join(self.role_mask_dir, f"{sample_id}_role_mask.npz")

    def _load_mask_np(self, sample_id):
        if not self.role_mask_dir or not sample_id:
            return None
        if sample_id in self._mask_cache:
            return self._mask_cache[sample_id]
        path = self._mask_path(sample_id)
        if not os.path.exists(path):
            self._mask_cache[sample_id] = None
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                masks = {
                    "onset_like": np.asarray(data["onset_like"], dtype=np.float32),
                    "nucleus_like": np.asarray(data["nucleus_like"], dtype=np.float32),
                    "transition_like": np.asarray(data["transition_like"], dtype=np.float32),
                    "valid_mask": np.asarray(data["valid_mask"], dtype=np.float32)
                    if "valid_mask" in data.files else None,
                }
        except Exception as exc:
            if self.strict_loading:
                raise ValueError(f"Invalid role mask {path}: {exc}") from exc
            masks = None
        self._mask_cache[sample_id] = masks
        return masks

    def _slice_mask(self, values, start_frame, num_frames, device):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 1 or values.size == 0:
            return None
        if self.crop_alignment == "fractional_linear":
            positions = float(start_frame) + np.arange(num_frames, dtype=np.float64)
            sliced = np.interp(
                positions,
                np.arange(values.size, dtype=np.float64),
                values,
                left=0.0,
                right=0.0,
            ).astype(np.float32, copy=False)
            tensor = torch.from_numpy(sliced).to(device=device, dtype=torch.float32)
            return torch.clamp(tensor, 0.0, 1.0)

        start_frame = int(start_frame)
        if start_frame < 0:
            values = np.pad(values, (abs(start_frame), 0), mode="constant")
            start_frame = 0
        sliced = values[start_frame:start_frame + num_frames]
        if sliced.size < num_frames:
            sliced = np.pad(sliced, (0, num_frames - sliced.size), mode="constant")
        tensor = torch.from_numpy(sliced[:num_frames]).to(device=device, dtype=torch.float32)
        return torch.clamp(tensor, 0.0, 1.0)

    def _build_batch_masks(self, meta, batch_size, num_frames, device):
        sample_ids = meta.get("sample_id", [""] * batch_size)
        starts = meta.get("start_sample", torch.zeros(batch_size, dtype=torch.long))
        padded_lefts = meta.get("padded_left", torch.zeros(batch_size, dtype=torch.long))
        if torch.is_tensor(starts):
            starts = starts.detach().cpu().tolist()
        if torch.is_tensor(padded_lefts):
            padded_lefts = padded_lefts.detach().cpu().tolist()

        onset, nucleus, transition, valid = [], [], [], []
        for idx in range(batch_size):
            sample_id = sample_ids[idx] if idx < len(sample_ids) else ""
            masks = self._load_mask_np(sample_id)
            if masks is None:
                zeros = torch.zeros(num_frames, device=device)
                onset.append(zeros)
                nucleus.append(zeros)
                transition.append(zeros)
                valid.append(torch.tensor(0.0, device=device))
                continue

            start_sample = int(starts[idx]) if idx < len(starts) else 0
            padded_left = int(padded_lefts[idx]) if idx < len(padded_lefts) else 0
            if self.crop_alignment == "fractional_linear":
                start_frame = (start_sample - padded_left) / float(self.hop_length)
            else:
                start_frame = int((start_sample - padded_left) // self.hop_length)
            onset_i = self._slice_mask(masks["onset_like"], start_frame, num_frames, device)
            nucleus_i = self._slice_mask(masks["nucleus_like"], start_frame, num_frames, device)
            transition_i = self._slice_mask(masks["transition_like"], start_frame, num_frames, device)
            if onset_i is None or nucleus_i is None or transition_i is None:
                zeros = torch.zeros(num_frames, device=device)
                onset.append(zeros)
                nucleus.append(zeros)
                transition.append(zeros)
                valid.append(torch.tensor(0.0, device=device))
                continue
            if masks.get("valid_mask") is not None:
                valid_i = self._slice_mask(masks["valid_mask"], start_frame, num_frames, device)
                active = valid_i is not None and torch.sum(valid_i) > self.eps
            else:
                active = (
                    torch.sum(onset_i) + torch.sum(nucleus_i) + torch.sum(transition_i)
                ) > self.eps
            onset.append(onset_i)
            nucleus.append(nucleus_i)
            transition.append(transition_i)
            valid.append(torch.tensor(1.0 if active else 0.0, device=device))

        return {
            "onset": torch.stack(onset, dim=0),
            "nucleus": torch.stack(nucleus, dim=0),
            "transition": torch.stack(transition, dim=0),
            "valid": torch.stack(valid, dim=0),
        }

    def _stft_mag(self, wav):
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        window = self.window.to(wav.device)
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            return_complex=True,
        )
        return torch.abs(spec).clamp_min(self.eps)

    def _masked_mean(self, values, mask):
        denom = torch.sum(mask).clamp_min(self.eps)
        return torch.sum(values * mask) / denom

    def _frequency_masks(self, num_freqs, device, dtype):
        cache = self._freqs
        if cache is None or cache.numel() != num_freqs or cache.device != device:
            cache = torch.linspace(
                0.0,
                self.sample_rate / 2.0,
                steps=num_freqs,
                device=device,
                dtype=dtype,
            )
            self._freqs = cache
        else:
            cache = cache.to(dtype=dtype)
        high_freq = (cache >= self.highband_start_hz).to(dtype).view(1, num_freqs, 1)
        nucleus_freq = (
            (cache >= self.nucleus_low_hz)
            & (cache <= min(self.nucleus_high_hz, self.sample_rate / 2.0))
        ).to(dtype).view(1, num_freqs, 1)
        return high_freq, nucleus_freq

    def _loss_from_magnitudes(self, est_mag, ref_mag, meta):
        batch_size, num_freqs, num_frames = est_mag.shape
        masks = self._build_batch_masks(meta, batch_size, num_frames, est_mag.device)
        valid_ratio = torch.mean(masks["valid"])
        if torch.sum(masks["valid"]) <= self.eps:
            zero = est_mag.sum() * 0.0
            stats = self._empty_stats(zero)
            stats["valid_ratio"] = valid_ratio.detach()
            stats["onset_coverage"] = masks["onset"].mean().detach()
            stats["nucleus_coverage"] = masks["nucleus"].mean().detach()
            stats["transition_coverage"] = masks["transition"].mean().detach()
            return zero, stats

        high_freq, nucleus_freq = self._frequency_masks(num_freqs, est_mag.device, est_mag.dtype)
        log_est = torch.log1p(est_mag)
        log_ref = torch.log1p(ref_mag)

        total = est_mag.sum() * 0.0
        stats = self._empty_stats(total)
        stats["onset_coverage"] = masks["onset"].mean().detach()
        stats["nucleus_coverage"] = masks["nucleus"].mean().detach()
        stats["transition_coverage"] = masks["transition"].mean().detach()

        if self.onset_weight > 0:
            onset_flux_est = torch.relu(log_est[:, :, 1:] - log_est[:, :, :-1])
            onset_flux_ref = torch.relu(log_ref[:, :, 1:] - log_ref[:, :, :-1])
            onset_mask = masks["onset"][:, 1:].unsqueeze(1) * high_freq * masks["valid"].view(-1, 1, 1)
            onset_loss = self._masked_mean(torch.abs(onset_flux_est - onset_flux_ref), onset_mask)
            total = total + self.onset_weight * onset_loss
            stats["onset_loss"] = onset_loss.detach()

        if self.nucleus_weight > 0:
            nucleus_mask = masks["nucleus"].unsqueeze(1) * nucleus_freq * masks["valid"].view(-1, 1, 1)
            nucleus_loss = self._masked_mean(torch.abs(log_est - log_ref), nucleus_mask)
            total = total + self.nucleus_weight * nucleus_loss
            stats["nucleus_loss"] = nucleus_loss.detach()

        if self.transition_weight > 0:
            delta_est = log_est[:, :, 1:] - log_est[:, :, :-1]
            delta_ref = log_ref[:, :, 1:] - log_ref[:, :, :-1]
            transition_mask = masks["transition"][:, 1:].unsqueeze(1) * masks["valid"].view(-1, 1, 1)
            transition_loss = self._masked_mean(torch.abs(delta_est - delta_ref), transition_mask)
            total = total + self.transition_weight * transition_loss
            stats["transition_loss"] = transition_loss.detach()

        stats["valid_ratio"] = valid_ratio.detach()
        return total, stats

    def forward_specs(self, enhanced_spec, clean_spec, meta):
        if meta is None:
            zero = enhanced_spec.abs().sum() * 0.0
            return zero, self._empty_stats(zero)
        if enhanced_spec.dim() == 4:
            enhanced_spec = enhanced_spec.squeeze(1)
        if clean_spec.dim() == 4:
            clean_spec = clean_spec.squeeze(1)
        frames = min(enhanced_spec.shape[-1], clean_spec.shape[-1])
        enhanced_spec = enhanced_spec[..., :frames]
        clean_spec = clean_spec[..., :frames]
        est_mag = enhanced_spec.abs().clamp_min(self.eps)
        ref_mag = clean_spec.abs().clamp_min(self.eps)
        return self._loss_from_magnitudes(est_mag, ref_mag, meta)

    def forward(self, enhanced_wav, clean_wav, meta):
        if meta is None:
            zero = enhanced_wav.sum() * 0.0
            return zero, self._empty_stats(zero)

        min_len = min(enhanced_wav.shape[-1], clean_wav.shape[-1])
        enhanced_wav = enhanced_wav[..., :min_len]
        clean_wav = clean_wav[..., :min_len]

        est_mag = self._stft_mag(enhanced_wav)
        ref_mag = self._stft_mag(clean_wav)
        return self._loss_from_magnitudes(est_mag, ref_mag, meta)

    def _empty_stats(self, zero):
        detached = zero.detach()
        return {
            "onset_loss": detached,
            "nucleus_loss": detached,
            "transition_loss": detached,
            "valid_ratio": detached,
            "onset_coverage": detached,
            "nucleus_coverage": detached,
            "transition_coverage": detached,
        }
