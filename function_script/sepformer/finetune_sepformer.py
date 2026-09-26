import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from pesq import pesq
from pystoi import stoi
from speechbrain.inference.separation import SepformerSeparation
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None


EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SepFormer on Tibetan speech + noise mixtures.")
    parser.add_argument("--train_dir", type=str, default="zang_data/train")
    parser.add_argument("--valid_dir", type=str, default="zang_data/val")
    parser.add_argument("--source", type=str, default="pretrained_models/sepformer-whamr",
                        help="SpeechBrain SepFormer source or local pretrained folder.")
    parser.add_argument("--savedir", type=str, default="pretrained_models/sepformer-whamr",
                        help="Local folder used by SpeechBrain for pretrained assets.")
    parser.add_argument("--output_dir", type=str, default="sepformer_tibetan_ft")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sample_rate", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--segment_seconds", type=float, default=4.0)
    parser.add_argument("--max_train_files", type=int, default=0)
    parser.add_argument("--max_valid_files", type=int, default=0)
    parser.add_argument("--num_eval_files", type=int, default=20,
                        help="Number of validation files used for PESQ/SI-SDR/ESTOI metrics. Use 0 to disable.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train_all", action="store_true",
                        help="Fine-tune encoder/decoder too. Default trains masknet only.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast mixed precision.")
    parser.add_argument("--nolog", action="store_true", help="Disable wandb logging.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def si_sdr(estimate, reference):
    estimate = estimate - estimate.mean(dim=1, keepdim=True)
    reference = reference - reference.mean(dim=1, keepdim=True)
    projection = (
        torch.sum(estimate * reference, dim=1, keepdim=True)
        * reference
        / (torch.sum(reference ** 2, dim=1, keepdim=True) + EPS)
    )
    noise = estimate - projection
    ratio = (torch.sum(projection ** 2, dim=1) + EPS) / (torch.sum(noise ** 2, dim=1) + EPS)
    return 10.0 * torch.log10(ratio)


def pit_si_sdr_loss(estimates, targets):
    # estimates/targets: [B, T, 2]
    est_0, est_1 = estimates[..., 0], estimates[..., 1]
    tgt_0, tgt_1 = targets[..., 0], targets[..., 1]
    score_a = 0.5 * (si_sdr(est_0, tgt_0) + si_sdr(est_1, tgt_1))
    score_b = 0.5 * (si_sdr(est_0, tgt_1) + si_sdr(est_1, tgt_0))
    return -torch.maximum(score_a, score_b).mean()


def safe_pesq(sample_rate, reference, estimate):
    try:
        mode = "nb" if sample_rate == 8000 else "wb"
        return float(pesq(sample_rate, reference, estimate, mode))
    except Exception:
        return float("nan")


def safe_estoi(sample_rate, reference, estimate):
    try:
        return float(stoi(reference, estimate, sample_rate, extended=True))
    except Exception:
        return float("nan")


def load_audio(path, sample_rate):
    path = str(path)
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)
    return wav.squeeze(0).float()


class TibetanSeparationDataset(Dataset):
    def __init__(self, root_dir, sample_rate, segment_samples, train=True, max_files=0):
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_samples)
        self.train = train
        self.mix_files = sorted(self.root_dir.glob("*_mix.wav"))
        if max_files and max_files > 0:
            self.mix_files = self.mix_files[:max_files]
        if not self.mix_files:
            raise RuntimeError(f"No *_mix.wav files found in {root_dir}")

    def __len__(self):
        return len(self.mix_files)

    def _crop_or_pad(self, mix, s1):
        min_len = min(mix.numel(), s1.numel())
        mix, s1 = mix[:min_len], s1[:min_len]
        target_len = self.segment_samples
        if min_len >= target_len:
            if self.train:
                start = random.randint(0, min_len - target_len)
            else:
                start = (min_len - target_len) // 2
            return mix[start:start + target_len], s1[start:start + target_len]
        pad = target_len - min_len
        return F.pad(mix, (0, pad)), F.pad(s1, (0, pad))

    def __getitem__(self, index):
        mix_path = self.mix_files[index]
        sample_id = mix_path.name.replace("_mix.wav", "")
        s1_path = self.root_dir / f"{sample_id}_source1.wav"
        if not s1_path.exists():
            raise FileNotFoundError(f"Missing clean source: {s1_path}")
        mix = load_audio(mix_path, self.sample_rate)
        s1 = load_audio(s1_path, self.sample_rate)
        mix, s1 = self._crop_or_pad(mix, s1)
        s2 = mix - s1
        norm = mix.abs().max().clamp_min(1e-4)
        mix = mix / norm
        s1 = s1 / norm
        s2 = s2 / norm
        targets = torch.stack([s1, s2], dim=-1)
        return mix, targets, sample_id


def separate_batch(model, mix):
    mix_w = model.mods.encoder(mix)
    est_mask = model.mods.masknet(mix_w)
    mix_w = torch.stack([mix_w] * model.hparams.num_spks)
    sep_h = mix_w * est_mask
    est_source = torch.cat(
        [model.mods.decoder(sep_h[i]).unsqueeze(-1) for i in range(model.hparams.num_spks)],
        dim=-1,
    )
    t_origin = mix.size(1)
    t_est = est_source.size(1)
    if t_origin > t_est:
        est_source = F.pad(est_source, (0, 0, 0, t_origin - t_est))
    else:
        est_source = est_source[:, :t_origin, :]
    return est_source


def configure_trainable(model, train_all):
    for param in model.mods.parameters():
        param.requires_grad = train_all
    if not train_all:
        for param in model.mods.masknet.parameters():
            param.requires_grad = True


def run_epoch(model, loader, optimizer, scaler, device, train, amp):
    model.mods.train(train)
    total_loss = 0.0
    total_items = 0
    iterator = tqdm(loader, desc="train" if train else "valid", leave=False)
    for mix, targets, _ in iterator:
        mix = mix.to(device)
        targets = targets.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", enabled=amp and str(device).startswith("cuda")):
                estimates = separate_batch(model, mix)
                loss = pit_si_sdr_loss(estimates, targets)
        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.mods.parameters() if p.requires_grad], 5.0
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.mods.parameters() if p.requires_grad], 5.0
                )
                optimizer.step()
        batch_size = mix.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
        iterator.set_postfix(loss=total_loss / max(total_items, 1))
    return total_loss / max(total_items, 1)


def select_speech_channel(estimates, reference):
    # estimates: [T, 2], reference: [T]
    min_len = min(estimates.size(0), reference.numel())
    estimates = estimates[:min_len]
    reference = reference[:min_len].unsqueeze(0)
    scores = []
    for idx in range(estimates.size(-1)):
        scores.append(si_sdr(estimates[:, idx].unsqueeze(0), reference).item())
    best_idx = int(np.argmax(scores))
    return estimates[:, best_idx], scores[best_idx]


def evaluate_metrics(model, dataset, device, sample_rate, num_eval_files):
    if num_eval_files == 0:
        return {"valid_si_sdr": float("nan"), "valid_pesq": float("nan"), "valid_estoi": float("nan")}
    model.mods.eval()
    limit = min(len(dataset), num_eval_files)
    si_sdr_scores, pesq_scores, estoi_scores = [], [], []
    with torch.no_grad():
        for idx in tqdm(range(limit), desc="metric", leave=False):
            mix, targets, _ = dataset[idx]
            mix = mix.unsqueeze(0).to(device)
            estimates = separate_batch(model, mix).squeeze(0).detach().cpu()
            reference = targets[:, 0].detach().cpu()
            selected, selected_si_sdr = select_speech_channel(estimates, reference)
            min_len = min(selected.numel(), reference.numel())
            selected_np = selected[:min_len].numpy()
            reference_np = reference[:min_len].numpy()
            si_sdr_scores.append(selected_si_sdr)
            pesq_scores.append(safe_pesq(sample_rate, reference_np, selected_np))
            estoi_scores.append(safe_estoi(sample_rate, reference_np, selected_np))
    return {
        "valid_si_sdr": float(np.nanmean(si_sdr_scores)),
        "valid_pesq": float(np.nanmean(pesq_scores)),
        "valid_estoi": float(np.nanmean(estoi_scores)),
    }


def save_checkpoint(model, path, epoch, valid_loss, metrics, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "valid_loss": valid_loss,
            "metrics": metrics,
            "encoder": model.mods.encoder.state_dict(),
            "masknet": model.mods.masknet.state_dict(),
            "decoder": model.mods.decoder.state_dict(),
            "args": vars(args),
        },
        path,
    )


def write_history(output_dir, history):
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    if history:
        with open(output_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    if not args.nolog:
        if wandb is None:
            print("Warning: wandb is not installed; continue without wandb logging.")
        else:
            wandb_run = wandb.init(project="sepformer-tibetan", config=vars(args), dir=str(output_dir))

    model = SepformerSeparation.from_hparams(
        source=args.source,
        savedir=args.savedir,
        run_opts={"device": str(device)},
    )
    configure_trainable(model, args.train_all)

    segment_samples = int(args.segment_seconds * args.sample_rate)
    train_set = TibetanSeparationDataset(
        args.train_dir, args.sample_rate, segment_samples, train=True, max_files=args.max_train_files
    )
    valid_set = TibetanSeparationDataset(
        args.valid_dir, args.sample_rate, segment_samples, train=False, max_files=args.max_valid_files
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False
    )
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False
    )

    params = [p for p in model.mods.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    history = []
    best_pesq = -float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scaler, device, train=True, amp=args.amp)
        valid_loss = run_epoch(model, valid_loader, optimizer, None, device, train=False, amp=False)
        metrics = evaluate_metrics(model, valid_set, device, args.sample_rate, args.num_eval_files)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            **metrics,
            "best_pesq": best_pesq if best_pesq > -float("inf") else float("nan"),
        }
        history.append(row)
        valid_pesq = metrics["valid_pesq"]
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} "
            f"valid_si_sdr={metrics['valid_si_sdr']:.4f} valid_pesq={valid_pesq:.4f} "
            f"valid_estoi={metrics['valid_estoi']:.4f}"
        )
        save_checkpoint(model, output_dir / "last.pt", epoch, valid_loss, metrics, args)
        if not np.isnan(valid_pesq) and valid_pesq > best_pesq:
            best_pesq = valid_pesq
            best_epoch = epoch
            save_checkpoint(model, output_dir / "best_pesq.pt", epoch, valid_loss, metrics, args)
        row["best_pesq"] = best_pesq
        row["best_epoch"] = best_epoch
        write_history(output_dir, history)
        if wandb_run is not None:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "valid/loss": valid_loss,
                "valid/si_sdr": metrics["valid_si_sdr"],
                "valid/pesq": metrics["valid_pesq"],
                "valid/estoi": metrics["valid_estoi"],
                "best/pesq": best_pesq,
                "best/epoch": best_epoch,
            }, step=epoch)
 
    if wandb_run is not None:
        wandb.finish()
    print(f"Best PESQ checkpoint: {output_dir / 'best_pesq.pt'}")
    print(f"Best epoch: {best_epoch}, best PESQ: {best_pesq:.4f}")


if __name__ == "__main__":
    main()
