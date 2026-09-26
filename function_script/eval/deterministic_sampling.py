"""Deterministic per-utterance RNG helpers for matched stochastic inference."""

import hashlib
from contextlib import contextmanager

import torch


SAMPLING_POLICY = "sample_id_sha256_v1"


def utterance_seed(base_seed, sample_id):
    payload = f"{int(base_seed)}\0{str(sample_id)}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % (2**63 - 1)


@contextmanager
def utterance_rng(base_seed, sample_id, device):
    device = torch.device(device)
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    seed = utterance_seed(base_seed, sample_id)
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        yield seed
