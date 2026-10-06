"""Frozen official DeepGaze III reference; all execution belongs in Slurm."""
import argparse
from collections import defaultdict
import bz2
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
from deepgaze_pytorch import DeepGazeIII


def predict(model, image, history, template):
    """History is chronological, in image pixels; absent older fixations are NaN."""
    h, w = image.shape[:2]
    history = np.asarray(history, dtype=np.float32)
    if history.ndim != 2 or history.shape[1] != 2 or not len(history):
        raise ValueError("At least one genuine (x,y) fixation is required")
    if not np.isfinite(history).all():
        raise ValueError("Nonfinite fixation history")
    selected = np.full((len(model.included_fixations), 2), np.nan, np.float32)
    for i, offset in enumerate(model.included_fixations):
        if len(history) >= -offset:
            selected[i] = history[offset]
    cb = zoom(template, (h / template.shape[0], w / template.shape[1]), order=0, mode="nearest")
    cb = (cb - logsumexp(cb)).astype(np.float32)
    tensor = lambda a: torch.as_tensor(a, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        logp = model(tensor(image.transpose(2, 0, 1)[None]), tensor(cb[None]),
                     tensor(selected[:, 0][None]), tensor(selected[:, 1][None]))
    logp = logp[0, 0].cpu().numpy()
    if logp.shape != (h, w) or not np.isfinite(logp).all() or abs(logsumexp(logp)) > 2e-5:
        raise AssertionError("Invalid or unnormalized next-fixation probability")
    return logp, cb


def plot(image, history, logp, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].imshow(image)
    axes[1].imshow(logp, cmap="magma")
    for ax in axes:
        ax.plot(*np.asarray(history).T, "o-", color="cyan", markersize=4)
        ax.scatter(*history[-1], c="yellow", s=65, edgecolors="black", zorder=5)
        ax.set_axis_off()
    axes[0].set_title("Image + past fixations (yellow = latest)")
    axes[1].set_title("Predicted next fixation: log probability")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["demo", "score"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, help="A's audited data manifest with fixed smoke_images")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--observers", type=int, default=2, help="First N subject IDs per image, fixed before scoring")
    p.add_argument("--steps", type=int, default=4, help="Maximum scored next fixations per trial")
    args = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run inside Slurm")
    started = time.monotonic()
    torch.manual_seed(20260928)
    np.random.seed(20260928)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 2)))
    model = DeepGazeIII(pretrained=True).eval().cuda()
    model.requires_grad_(False)
    template = np.load(args.root / "centerbias.npy")
    meta = dict(mode=args.mode, job=os.environ["SLURM_JOB_ID"], seed=20260928, argv=sys.argv,
                started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                upstream_commit=subprocess.check_output(["git", "-C", str(args.root / "vendor"), "rev-parse", "HEAD"], text=True).strip(),
                source_sha256=digest(__file__), checkpoint_sha256=digest(args.root / "cache/torch/hub/checkpoints/deepgaze3.pth"),
                centerbias="Official MIT1003 template; no COCO fitting", centerbias_sha256=digest(args.root / "centerbias.npy"),
                device=torch.cuda.get_device_name(), torch=torch.__version__, initial_fixation_scored=False,
                numpy=np.__version__, pillow=Image.__version__, project_commit=None,
                project_commit_note="No project Git repository; source snapshot and SHA256 retained")
    if args.mode == "demo":
        image = np.frombuffer(bz2.decompress((args.root / "face.dat.bz2").read_bytes()), dtype=np.uint8).reshape(768, 1024, 3).copy()
        history = np.array([[512, 384], [300, 300], [500, 100], [200, 300], [200, 100], [700, 500]], np.float32)
        logp, _ = predict(model, image, history, template)
        checks = []
        for length in [1, 2, 3, 4]:
            short, _ = predict(model, image, history[:length], template)
            checks.append(dict(history_length=length, probability_sum=float(np.exp(short.astype(np.float64)).sum())))
        repeat, _ = predict(model, image, history, template)
        assert np.allclose(logp, repeat, atol=1e-6, rtol=0)
        changed, _ = predict(model, image, history[:1], template)
        assert not np.allclose(logp, changed)
        np.savez_compressed(args.output / "prediction.npz", log_probability=logp, history_xy=history)
        plot(image, history, logp, args.output / "demo.png")
        meta.update(probability_sum=float(np.exp(logp.astype(np.float64)).sum()), short_history_checks=checks,
                    repeat_max_abs_difference=float(np.max(np.abs(logp - repeat))), image_size=[1024, 768],
                    history_is_synthetic=True, description="Official README example; not a human-data score")
    else:
        if not args.manifest or min(args.limit, args.observers, args.steps) < 1:
            p.error("score requires --manifest and positive limits")
        manifest_bytes = args.manifest.read_bytes()
        manifest = json.loads(manifest_bytes)
        data = json.loads(Path(manifest["raw_fixations"]).read_text())
        names = manifest["smoke_images"][:args.limit]
        rows = [r for r in data if r["name"] in names]
        if {r["name"] for r in rows} != set(names) or any(r["split"] != "train" or r["condition"] != "freeview" for r in rows):
            raise ValueError("Smoke scoring must use named official training/freeview images only")
        results, masses, skipped, image_hashes, trials = [], [], [], {}, 0
        assert manifest["display_size"] == [1680, 1050]
        for name in names:
            entry = manifest["images"][name]
            assert entry["split"] == "train" and entry["smoke"]
            image_hashes[name] = digest(entry["path"])
            assert image_hashes[name] == entry["sha256"]
            with Image.open(entry["path"]) as source:
                image = np.array(source.convert("RGB"))
            if image.shape[:2] != (1050, 1680):
                raise ValueError("Expected official already-padded 1680x1050 stimuli; do not guess geometry")
            selected_rows = sorted((r for r in rows if r["name"] == name), key=lambda r:r["subject"])[:args.observers]
            trials += len(selected_rows)
            for row in selected_rows:
                xy = np.column_stack([row["X"], row["Y"]]).astype(np.float32)
                for t in range(1, min(len(xy), args.steps+1)):
                    if not np.isfinite(xy[:t+1]).all() or not (0 <= xy[t, 0] < 1680 and 0 <= xy[t, 1] < 1050):
                        skipped.append(dict(image=name, subject=row["subject"], fixation_index=t, reason="nonfinite prefix or target outside canvas"))
                        continue
                    x, y = np.floor(xy[t]).astype(int)
                    logp, cb = predict(model, image, xy[:t], template)
                    results.append(dict(image=name, subject=row["subject"], fixation_index=t, x=float(xy[t, 0]), y=float(xy[t, 1]),
                                        pixel_logp=float(logp[y, x]), centerbias_pixel_logp=float(cb[y, x])))
                    masses.append(float(np.exp(logp.astype(np.float64)).sum()))
                    if len(results) == 1:
                        plot(image, xy[:t], logp, args.output / "freeview.png")
                        np.savez_compressed(args.output / "prediction.npz", log_probability=logp, history_xy=xy[:t])
            print(f"Scored {name}; n={len(results)}", flush=True)
        if not results:
            raise ValueError("No eligible next fixations")
        with (args.output / "scores.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        by_image = defaultdict(list)
        for r in results:
            by_image[r["image"]].append(r["pixel_logp"])
        meta.update(images=names, image_sha256=image_hashes, trials=trials, predictions=len(results), split="A's official-train smoke subset",
                    fixation_file_sha256=digest(manifest["raw_fixations"]), manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                    preprocessing="Official already-padded 1680x1050 RGB images unchanged; native display coordinates; floor for pixel lookup",
                    scored_fixation_indices=f"1..{args.steps} (0-based); first {args.observers} subject IDs per image; initial index 0 conditions only",
                    mean_pixel_log_likelihood_nats=float(np.mean([r["pixel_logp"] for r in results])),
                    image_mean_pixel_log_likelihood_nats={n:float(np.mean(v)) for n,v in by_image.items()},
                    equal_image_mean_pixel_log_likelihood_nats=float(np.mean([np.mean(v) for v in by_image.values()])),
                    mean_bits_over_centerbias=float(np.mean([r["pixel_logp"]-r["centerbias_pixel_logp"] for r in results])/np.log(2)),
                    probability_sum_range=[min(masses), max(masses)], skipped=skipped)
    meta.update(elapsed_seconds=time.monotonic()-started, peak_cuda_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
    (args.output / "summary.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
