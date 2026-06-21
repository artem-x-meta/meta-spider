"""Export test prompts from the v9 dataset.pt into the format for llama-extract-activations.

Takes the last test_size samples (the same split as Phase 1R: train+val then test),
writes:
  - prompts.bin   — input_text of each prompt, separated by a \0 byte (for the C++ tool)
  - fp16_ref.npz  — fp16 activations [N, n_layers, hidden] from dataset.pt (sanity-gate reference)
  - meta.json     — layers, n, hidden, order

Usage:
    python export_prompts.py --dataset <v9 dataset.pt> --out-dir <results>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# Phase 1R split: train 5000 / val 500 / test 500 (last 500 of 6000)
TRAIN, VAL, TEST = 5000, 500, 500
TARGET_LAYERS = [10, 14, 18, 22, 25]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--test-size", type=int, default=TEST)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.dataset, weights_only=False, map_location="cpu")
    samples = payload["samples"]
    test = samples[TRAIN + VAL: TRAIN + VAL + args.test_size]
    print(f"test samples: {len(test)}")

    # Prompts → \0-separated bin
    prompts = [s["input_text"] for s in test]
    blob = b"\x00".join(p.encode("utf-8") for p in prompts)
    (out / "prompts.bin").write_bytes(blob)

    # fp16 reference activations [N, n_layers, hidden]
    layers = sorted(test[0]["activations"].keys())
    assert layers == TARGET_LAYERS, f"layers {layers} != {TARGET_LAYERS}"
    hidden = test[0]["activations"][layers[0]].shape[-1]
    ref = np.zeros((len(test), len(layers), hidden), dtype=np.float32)
    for i, s in enumerate(test):
        for j, L in enumerate(layers):
            ref[i, j] = s["activations"][L].float().numpy()
    np.savez_compressed(out / "fp16_ref.npz", activations=ref, layers=np.array(layers))

    # Metadata + answers/correctness for offline_curve
    meta = {
        "n": len(test),
        "layers": layers,
        "hidden": int(hidden),
        "pass1_correct": [bool(s["pass1_correct"]) for s in test],
        "ground_truth": [s["ground_truth"] for s in test],
        "aliases": [s.get("aliases") for s in test],
        "input_text": prompts,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    print(f"wrote prompts.bin ({len(blob)} bytes), fp16_ref.npz {ref.shape}, meta.json")
    print(f"layers={layers}, hidden={hidden}")


if __name__ == "__main__":
    main()
