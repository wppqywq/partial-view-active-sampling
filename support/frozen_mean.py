"""Mean-only predictor from scratch: weighted MSE, no residual or variance head."""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time

if not os.environ.get("SLURM_JOB_ID"):
    raise RuntimeError("Run inside the supplied Slurm job")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import frozen_data as P

SEED = 20260928
UPSTREAM = Path("/mnt/disk2/youyouyang/proposal2/predictor_d")
V1 = UPSTREAM / "runs/22351"
NORMALIZATION_HASH = "9c07ab5b54147987125ecc68f39382ec22c27dd5b4205e2e67ef253601f9ad6b"
DATA_SOURCE_HASH = "0b48aa2db43e20036362f306a8f4838bd36eb3d666f3160ecb28e0bcfde54d15"
OBSERVER_HASH = "c3eb13cfb186cf6472f44c5c9ed970411bfcafae3c06bf3ae8779eb5a4fd1361"
WEIGHTS_HASH = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
STOP = False


def request_stop(*_):
    global STOP
    STOP = True


signal.signal(signal.SIGUSR1, request_stop)
signal.signal(signal.SIGTERM, request_stop)


class DirectMean(nn.Module):
    """State keys: body.{0,2,4}.{weight,bias}, out.{weight,bias}."""
    def __init__(self, channels=384):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels + 4, 128, 1), nn.GELU(),
                                  nn.Conv2d(128, 128, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(128, 128, 3, padding=2, dilation=2), nn.GELU())
        self.out = nn.Conv2d(128, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, features, reveal, weight):
        b, _, h, w = features.shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=features.device),
                                torch.linspace(-1, 1, w, device=features.device), indexing="ij")
        xy = torch.stack((xx, yy))[None].expand(b, -1, -1, -1)
        return self.out(self.body(torch.cat((features, reveal, weight, xy), 1)))


class FrozenEncoder(nn.Module):
    """RGB BCHW -> raw DINO features B384x20x32; caller normalizes channels."""
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        weights = UPSTREAM / "cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"
        repo = UPSTREAM / "vendor/dinov2"
        assert P.digest(weights) == WEIGHTS_HASH
        assert subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() == REVISION
        self.model = torch.hub.load(str(repo), "dinov2_vits14", source="local", pretrained=False)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model.to(self.device).eval().requires_grad_(False)
        self.register_buffer("rgb_mean", torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None])
        self.register_buffer("rgb_std", torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None])
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, rgb):
        assert rgb.ndim == 4 and rgb.shape[1:] == (3, 280, 448)
        rgb = rgb.to(self.device)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            value = self.model.forward_features((rgb - self.rgb_mean) / self.rgb_std)["x_norm_patchtokens"]
        value = value.transpose(1, 2).reshape(-1, 384, 20, 32).float()
        assert torch.isfinite(value).all()
        return value


def load_frozen(device="cuda"):
    return FrozenEncoder(device)


def mse_per_image(predicted, target, weight):
    error = (predicted - target).square().mean(1)
    mass = weight[:, 0]
    return (error * mass).sum((1, 2)) / mass.sum((1, 2)).clamp_min(1e-9)


@torch.no_grad()
def evaluate(model, names, cache, mean, std, path):
    model.eval()
    records = []
    for name in names:
        partial = torch.load(cache / "development" / (name + ".pt"), weights_only=True)
        target = torch.load(cache / "targets" / (name + ".pt"), weights_only=True)
        features = (partial["features"].cuda().float() - mean) / std
        truth = (target["features"][None].cuda().float() - mean) / std
        weight = target["weight"][None].cuda().expand(len(features), -1, -1, -1)
        predicted = model(features, partial["reveal"].cuda(), weight)
        values = {"direct_mse": mse_per_image(predicted, truth, weight),
                  "raw_feature_mse": mse_per_image(features, truth, weight),
                  "zero_mse": mse_per_image(torch.zeros_like(features), truth, weight)}
        assert all(torch.isfinite(v).all() for v in values.values())
        for i, row in enumerate(partial["records"]):
            records.append({**{k: v for k, v in row.items() if k != "xy"},
                            **{key: array[i].item() for key, array in values.items()}})
    keys = ("direct_mse", "raw_feature_mse", "zero_mse")
    images = defaultdict(list)
    for row in records:
        images[row["image"]].append(row)
    image_rows = [{"image": name, **{k: float(np.mean([r[k] for r in rows])) for k in keys}}
                  for name, rows in images.items()]
    overall = {k: float(np.mean([r[k] for r in image_rows])) for k in keys}
    strata = []
    for kind in ("human", "random"):
        for budget in (1, 4, 8):
            selected = [r for r in records if r["kind"] == kind and r["budget"] == budget]
            strata.append({"kind": kind, "budget": budget, "images": len(selected),
                           "short_histories": sum(r["actual_budget"] < budget for r in selected),
                           **{k: float(np.mean([r[k] for r in selected])) for k in keys}})
    paired = []
    for kind in ("human", "random"):
        lookup = {(r["image"], r["budget"]): r for r in records if r["kind"] == kind}
        for before, after in ((1, 4), (4, 8), (1, 8)):
            gains = [lookup[(name, before)]["direct_mse"] - lookup[(name, after)]["direct_mse"] for name in names]
            paired.append({"kind": kind, "from_budget": before, "to_budget": after,
                           "mean_mse_reduction": float(np.mean(gains)), "images_improved_fraction": float(np.mean(np.array(gains) > 0))})
    summary = {"images": len(names), "states": len(records), "overall_image_macro": overall,
               "by_kind_budget": strata, "paired_same_history": paired,
               "units": "train-standardized DINO features; content-pixel-fraction weighted; zero means train channel mean"}
    P.write_json(path, {"summary": summary, "images": image_rows, "states": records})
    return summary


def cpu_copy(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: cpu_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cpu_copy(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(cpu_copy(v) for v in obj)
    return copy.deepcopy(obj)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    torch.cuda.set_rng_state_all(value["cuda"])


def accumulator():
    return {"mse_sum": 0., "images": 0, "state_counts": {}}


def main(args):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    run = Path(args.run_dir)
    assert P.digest(Path(__file__).with_name("frozen_data.py")) == DATA_SOURCE_HASH
    assert P.digest(Path(__file__).with_name("observer.py")) == OBSERVER_HASH
    assert P.digest(V1 / "normalization.pt") == NORMALIZATION_HASH
    upstream = json.loads((V1 / "config.json").read_text())
    manifest_path = Path("/mnt/disk2/youyouyang/proposal2/coco_freeview/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert P.digest(manifest_path) == upstream["manifest_sha256"]
    assert P.digest(manifest["raw_fixations"]) == upstream["fixations_sha256"]
    entries = {n: e for n, e in manifest["images"].items() if e["split"] in ("train", "development")}
    train_names = sorted(n for n, e in entries.items() if e["split"] == "train")
    dev_names = sorted(n for n, e in entries.items() if e["split"] == "development")
    assert len(train_names) == 3343 and len(dev_names) == 371
    histories = defaultdict(list)
    for row in json.loads(Path(manifest["raw_fixations"]).read_text()):
        if row["name"] in entries:
            histories[row["name"]].append(row)
    cache = Path(upstream["cache"])
    assert cache.name == "73d7ae38f654227711f1" and P.digest(cache / "normalization.pt") == NORMALIZATION_HASH
    for name in train_names + dev_names:
        assert (cache / "targets" / (name + ".pt")).is_file()
    for name in dev_names:
        assert (cache / "development" / (name + ".pt")).is_file()
    norm = torch.load(V1 / "normalization.pt", weights_only=True)
    assert norm["train_names"] == train_names
    shutil.copyfile(V1 / "normalization.pt", run / "normalization.pt")
    shutil.copyfile(V1 / "development_states.json", run / "development_states.json")
    specification = {"seed": SEED, "model": "DirectMean", "initialization": "from scratch; final mean weights/bias zero",
                     "model_state_keys": "body.{0,2,4}.{weight,bias}, out.{weight,bias}", "body_channels": 128,
                     "residual": False, "variance_head": False, "loss": "content weighted MSE",
                     "maximum_epochs": 60, "minimum_epochs": 10, "batch_size": 8,
                     "optimizer": {"name": "AdamW", "lr": .001, "weight_decay": .0001, "gradient_clip_norm": 1},
                     "scheduler": {"name": "ReduceLROnPlateau", "factor": .5, "patience": 3,
                                   "threshold": 1e-4, "threshold_mode": "rel", "min_lr": 1e-5},
                     "early_stop": {"patience": 10, "relative_improvement": 1e-4,
                                    "reference": "best significant improvement anchor; cumulative small improvements may reset counter"},
                     "training_source_sha256": P.digest(__file__), "data_source_sha256": DATA_SOURCE_HASH,
                     "observer_sha256": OBSERVER_HASH, "normalization_sha256": NORMALIZATION_HASH,
                     "revision": REVISION, "weights_sha256": WEIGHTS_HASH, "manifest_path": str(manifest_path),
                     "manifest_sha256": upstream["manifest_sha256"], "fixations_sha256": upstream["fixations_sha256"],
                     "cache": str(cache), "feature_cache": str(cache), "train_images": 3343,
                     "development_images": 371, "holdout_images_used": 0,
                     "torch": str(torch.__version__), "cuda": torch.version.cuda}
    experiment_id = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    config = {**specification, "experiment_id": experiment_id, "state": "initializing",
              "gpu": torch.cuda.get_device_name(), "best_checkpoint_sha256": None,
              "submit_source_sha256": P.digest(Path(__file__).with_name("run.sh")),
              "diagnostic_source_sha256": P.digest(Path(__file__).with_name("patch_diagnostics.py"))}
    P.write_json(run / "config.json", config)
    encoder = load_frozen()
    torch.manual_seed(SEED)  # Head initialization independent of encoder construction.
    model = DirectMean().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=3,
                                                         threshold=1e-4, threshold_mode="rel", min_lr=1e-5)
    first_epoch, first_batch, best, significant_best, bad_epochs = 0, 0, float("inf"), float("inf"), 0
    best_payload, best_summary, acc, history = None, None, accumulator(), []
    prior_status = "training"
    if args.resume:
        loaded = torch.load(args.resume, map_location="cpu", weights_only=False)
        assert loaded["experiment_id"] == experiment_id, "Resume training input/config/source mismatch"
        model.load_state_dict(loaded["model"])
        optimizer.load_state_dict(loaded["optimizer"])
        scheduler.load_state_dict(loaded["scheduler"])
        first_epoch, first_batch = loaded["epoch"], loaded["batch"]
        best, significant_best, bad_epochs = loaded["best"], loaded["significant_best"], loaded["bad_epochs"]
        best_payload = loaded.get("best_snapshot", loaded)
        best_summary, acc, history = loaded["best_metrics"], loaded["accumulator"], loaded["history"]
        prior_status = loaded["stop_status"]
        restore_rng(loaded["rng"])
        if best_payload is not None:
            assert best_payload["dev_mse"] == best
            P.save(run / "best.pt", best_payload)
            P.write_json(run / "best_metrics.json", best_summary)
    else:
        assert torch.count_nonzero(model.out.weight) == 0 and torch.count_nonzero(model.out.bias) == 0
    assert not encoder.training and all(not p.requires_grad for p in encoder.parameters())
    assert all(p.requires_grad for p in model.parameters())
    mean = norm["mean"].cuda()[None, :, None, None]
    std = norm["std"].cuda()[None, :, None, None]
    with open(run / "curves.jsonl", "w") as stream:
        for item in history:
            stream.write(json.dumps(item, allow_nan=False) + "\n")
    started = time.monotonic()

    def payload(epoch, batch, status, current_acc):
        return {"model": cpu_copy(model.state_dict()), "optimizer": cpu_copy(optimizer.state_dict()),
                "scheduler": scheduler.state_dict(), "rng": rng_state(), "epoch": epoch, "batch": batch,
                "best": best, "significant_best": significant_best, "bad_epochs": bad_epochs,
                "accumulator": cpu_copy(current_acc), "normalization": norm, "experiment_id": experiment_id,
                "stop_status": status, "history": copy.deepcopy(history), "best_metrics": best_summary, "config": dict(config)}

    def save_last(epoch, batch, status, current_acc):
        value = payload(epoch, batch, status, current_acc)
        value["best_snapshot"] = best_payload
        P.save(run / "last.pt", value)

    def finish(status, epoch):
        config.update(state=status, completed_epochs=epoch, bad_epochs=bad_epochs,
                      best_epoch=best_payload["epoch"] if best_payload is not None else None,
                      best_dev_mse=best if np.isfinite(best) else None,
                      best_checkpoint_sha256=P.digest(run / "best.pt") if best_payload is not None else None,
                      last_checkpoint_sha256=P.digest(run / "last.pt"),
                      current_lr=max(g["lr"] for g in optimizer.param_groups),
                      cuda_peak_memory_bytes=torch.cuda.max_memory_allocated(),
                      seconds_this_process=time.monotonic() - started)
        P.write_json(run / "config.json", config)

    if prior_status in ("early_stopped", "budget_exhausted"):
        save_last(first_epoch, first_batch, prior_status, acc)
        finish(prior_status, first_epoch)
        return 0
    config["state"] = "training"
    P.write_json(run / "config.json", config)
    for epoch in range(first_epoch, 60):
        names = list(np.asarray(train_names)[P.rng_for("order", epoch).permutation(len(train_names))])
        dataset = P.TrainingStates(names, entries, histories, cache / "targets", epoch)
        loader = DataLoader(dataset, batch_size=8, num_workers=2, pin_memory=True,
                            generator=torch.Generator().manual_seed(SEED + epoch))
        model.train()
        if epoch != first_epoch:
            acc = accumulator()
        state_counts = defaultdict(int, acc["state_counts"])
        lr_used = max(g["lr"] for g in optimizer.param_groups)
        for batch_index, batch in enumerate(loader):
            if epoch == first_epoch and batch_index < first_batch:
                continue
            features = (encoder(batch["rgb"]) - mean) / std
            target = (batch["target"].cuda() - mean) / std
            weight, reveal = batch["weight"].cuda(), batch["reveal"].cuda()
            predicted = model(features, reveal, weight)
            errors = mse_per_image(predicted, target, weight)
            assert predicted.shape == target.shape and torch.isfinite(errors).all()
            loss = errors.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            acc["mse_sum"] += errors.detach().sum().item()
            acc["images"] += len(features)
            for kind, budget, actual in zip(batch["kind"], batch["budget"], batch["actual_budget"]):
                state_counts[f"{kind}:requested={budget.item()}:actual={actual.item()}"] += 1
            acc["state_counts"] = dict(state_counts)
            if (batch_index + 1) % 100 == 0 or STOP:
                save_last(epoch, batch_index + 1, "training", acc)
                print(json.dumps({"stage": "train", "epoch": epoch + 1, "batch": batch_index + 1,
                                  "train_mse": acc["mse_sum"] / acc["images"], "lr": lr_used}), flush=True)
            if STOP:
                finish("interrupted_resumable", epoch)
                return 75
        assert acc["images"] == 3343
        assert {int(key.split(":requested=")[1].split(":")[0]) for key in state_counts} == set(range(1, 9))
        summary = evaluate(model, dev_names, cache, mean, std, run / f"development_epoch_{epoch + 1:03d}.json")
        dev_mse = summary["overall_image_macro"]["direct_mse"]
        relative_improvement = (significant_best - dev_mse) / abs(significant_best) if np.isfinite(significant_best) else None
        if relative_improvement is None or relative_improvement >= 1e-4:
            significant_best, bad_epochs = dev_mse, 0
        else:
            bad_epochs += 1
        improved = dev_mse < best
        if improved:
            best, best_summary = dev_mse, summary
        scheduler.step(dev_mse)
        status = "early_stopped" if epoch + 1 >= 10 and bad_epochs >= 10 else "budget_exhausted" if epoch + 1 == 60 else "training"
        record = {"epoch": epoch + 1, "train_mse": acc["mse_sum"] / acc["images"], "train_images": acc["images"],
                  "development_mse": dev_mse, "best_development_mse": best, "lr_used": lr_used,
                  "lr_after_scheduler": max(g["lr"] for g in optimizer.param_groups),
                  "relative_improvement_over_significant_anchor": relative_improvement,
                  "bad_epochs": bad_epochs, "stop_status": status,
                  "training_state_counts": dict(state_counts), "development": summary,
                  "train_metric_scope": "online pre-update full-epoch MSE; accumulator preserved across resume"}
        history.append(record)
        with open(run / "curves.jsonl", "a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if improved:
            best_payload = payload(epoch + 1, 0, status, accumulator())
            best_payload["dev_mse"] = best
            P.save(run / "best.pt", best_payload)
            P.write_json(run / "best_metrics.json", best_summary)
        save_last(epoch + 1, 0, status, accumulator())
        finish(status, epoch + 1)
        print(json.dumps({"stage": "epoch_complete", "epoch": epoch + 1, "train_mse": record["train_mse"],
                          "development_mse": dev_mse, "best_development_mse": best,
                          "lr": record["lr_after_scheduler"], "bad_epochs": bad_epochs, "stop_status": status}), flush=True)
        if status in ("early_stopped", "budget_exhausted"):
            return 0
        if STOP:
            finish("interrupted_resumable", epoch + 1)
            return 75
    raise RuntimeError("No training/terminal checkpoint state reached")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    try:
        exit_code = main(args)
    except BaseException as error:
        P.write_json(Path(args.run_dir) / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    raise SystemExit(exit_code)
