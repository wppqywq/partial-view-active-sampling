"""Full-data frozen DINOv2 partial-view predictor; execute ONLY inside Slurm."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import shutil
import time
from collections import defaultdict

if not os.environ.get("SLURM_JOB_ID"):
    raise RuntimeError("Use the supplied Slurm job")

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from observer import PartialObserver

SEED = 20260928
STOP = False


def request_stop(*_):
    global STOP
    STOP = True


signal.signal(signal.SIGUSR1, request_stop)
signal.signal(signal.SIGTERM, request_stop)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save(path, obj):
    temporary = Path(str(path) + ".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)


def rng_for(*parts):
    key = ":".join(map(str, (SEED,) + parts))
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))


def observer(entry):
    # Opening images is restricted to entries already selected as train/development.
    assert entry["split"] in ("train", "development")
    with Image.open(entry["path"]) as image:
        return PartialObserver(image, content_box=entry["display_content_box"])


def rgb_tensor(rgb):
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))


def pool_mask(mask):
    value = torch.from_numpy(mask.astype(np.float32))[None, None]
    return F.avg_pool2d(value, 14, 14)[0]


def history_for(obs, rows, rng, kind, budget):
    eligible = [row for row in rows if len(row["X"]) >= budget]
    row = (eligible or sorted(rows, key=lambda r: -len(r["X"])))
    row = row[int(rng.integers(len(row)))] if eligible else row[0]
    if kind == "human":
        points = np.column_stack((row["X"], row["Y"]))[:budget].tolist()
    else:
        points = [[row["X"][0], row["Y"][0]]]
        xy = obs.choice_grid()["xy"]
        previous = obs.observe(points)["reveal_mask"]
        # Uniform random permutation + rejection is uniform among currently
        # eligible candidates, avoiding an expensive all-candidate rescan.
        for candidate in rng.permutation(len(xy)):
            if len(points) >= budget:
                break
            proposal = xy[candidate].tolist()
            revealed = obs.observe(points + [proposal])["reveal_mask"]
            if np.any(revealed & ~previous):
                points.append(proposal)
                previous = revealed
    return points, {"subject": row["subject"], "actual_budget": len(points)}


class TrainingStates(Dataset):
    def __init__(self, names, entries, histories, cache, epoch):
        self.names, self.entries, self.histories = names, entries, histories
        self.cache, self.epoch = cache, epoch

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        rng = rng_for("train", self.epoch, name)
        budget = int(rng.integers(1, 9))
        kind = "human" if rng.random() < .5 else "random"
        obs = observer(self.entries[name])
        points, info = history_for(obs, self.histories[name], rng, kind, budget)
        view = obs.observe(points)
        target = torch.load(self.cache / (name + ".pt"), weights_only=True)
        return {"rgb": rgb_tensor(view["rgb"]),
                "reveal": pool_mask(view["reveal_mask"]),
                "weight": target["weight"].float(), "target": target["features"].float(),
                "budget": budget, "actual_budget": info["actual_budget"], "kind": kind}


class Predictor(nn.Module):
    def __init__(self, channels=384):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels + 4, 128, 1), nn.GELU(),
                                  nn.Conv2d(128, 128, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(128, 128, 3, padding=2, dilation=2), nn.GELU())
        self.out = nn.Conv2d(128, channels * 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, features, reveal, weight):
        b, _, h, w = features.shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=features.device),
                                torch.linspace(-1, 1, w, device=features.device), indexing="ij")
        xy = torch.stack((xx, yy))[None].expand(b, -1, -1, -1)
        delta, logvar = self.out(self.body(torch.cat((features, reveal, weight, xy), 1))).chunk(2, 1)
        return features + delta, logvar.clamp(-6, 4)


class Encoder:
    def __init__(self, repo):
        self.model = torch.hub.load(str(repo), "dinov2_vits14", source="local", pretrained=True)
        self.model.eval().requires_grad_(False).cuda()
        self.mean = torch.tensor([.485, .456, .406], device="cuda")[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device="cuda")[None, :, None, None]

    @torch.no_grad()
    def __call__(self, rgb):
        rgb = rgb.cuda(non_blocking=True)
        assert rgb.ndim == 4 and rgb.shape[1:] == (3, 280, 448)
        with torch.autocast("cuda", dtype=torch.float16):
            features = self.model.forward_features((rgb - self.mean) / self.std)["x_norm_patchtokens"]
        result = features.transpose(1, 2).reshape(-1, 384, 20, 32).float()
        assert torch.isfinite(result).all()
        return result


def prepare_cache(entries, train_names, dev_names, histories, enc, cache, run):
    targets = cache / "targets"
    dev = cache / "development"
    targets.mkdir(parents=True, exist_ok=True)
    dev.mkdir(parents=True, exist_ok=True)
    channel_sum = torch.zeros(384, dtype=torch.float64)
    channel_sq = channel_sum.clone()
    mass = 0.
    for index, name in enumerate(train_names + dev_names):
        path = targets / (name + ".pt")
        if not path.exists():
            assert digest(entries[name]["path"]) == entries[name]["sha256"], "Input image hash changed"
            obs = observer(entries[name])
            view = obs.observe([])
            if index == 0:
                assert view["rgb"].shape == obs.target().shape
                assert not view["reveal_mask"].any()
                assert not (view["valid_mask"] & ~view["display_mask"]).any()
                border = view["display_mask"] & ~view["valid_mask"]
                assert np.array_equal(view["rgb"][border], obs.target()[border])
                checked = obs.observe([[histories[name][0]["X"][0], histories[name][0]["Y"][0]]])
                assert not (checked["reveal_mask"] & ~checked["valid_mask"]).any()
                grid = obs.choice_grid()
                assert np.all(np.diff(grid["x_edges"]) > 0) and np.all(np.diff(grid["y_edges"]) > 0)
                assert (grid["x_edges"][0], grid["y_edges"][0], grid["x_edges"][-1], grid["y_edges"][-1]) == obs.content_box
            feat = enc(rgb_tensor(obs.target())[None])[0].cpu().half()
            weight = pool_mask(view["valid_mask"])
            assert weight.sum() > 0
            save(path, {"features": feat, "weight": weight})
        if name in train_names_set:
            payload = torch.load(path, weights_only=True)
            feat, weight = payload["features"].double(), payload["weight"].double()
            channel_sum += (feat * weight).sum((1, 2))
            channel_sq += (feat.square() * weight).sum((1, 2))
            mass += weight.sum().item()
        if index % 200 == 0:
            print(json.dumps({"stage": "target_cache", "images": index + 1, "total": len(entries)}), flush=True)
        if STOP:
            raise InterruptedError("Stopped during resumable target cache")
    mean = channel_sum / mass
    std = (channel_sq / mass - mean.square()).clamp_min(1e-6).sqrt()
    norm = {"mean": mean.float(), "std": std.float(), "train_names": train_names,
            "weight_mass": mass, "source": "train content-weighted full-target patch features only"}
    save(cache / "normalization.pt", norm)
    state_rows = []
    for index, name in enumerate(dev_names):
        path = dev / (name + ".pt")
        if not path.exists():
            obs = observer(entries[name])
            partials, reveals, records = [], [], []
            for kind in ("human", "random"):
                points, info = history_for(obs, histories[name], rng_for("development", name, kind), kind, 8)
                for budget in (1, 4, 8):
                    prefix = points[:budget]
                    view = obs.observe(prefix)
                    partials.append(rgb_tensor(view["rgb"]))
                    reveals.append(pool_mask(view["reveal_mask"]))
                    records.append({"image": name, "kind": kind, "budget": budget,
                                    "actual_budget": len(prefix), "subject": info["subject"], "xy": prefix})
            features = enc(torch.stack(partials)).cpu().half()
            save(path, {"features": features, "reveal": torch.stack(reveals), "records": records})
        state_rows.extend(torch.load(path, weights_only=True)["records"])
        if index % 50 == 0:
            print(json.dumps({"stage": "development_cache", "images": index + 1}), flush=True)
        if STOP:
            raise InterruptedError("Stopped during resumable development cache")
    write_json(run / "development_states.json", state_rows)
    return norm


def metrics(mean, logvar, target, weight):
    error = (mean - target).square()
    variance = logvar.exp()
    nll = .5 * (math.log(2 * math.pi) + logvar + error / variance)
    denominator = weight.sum((1, 2, 3)) * mean.shape[1]
    reduce = lambda array: (array * weight).sum((1, 2, 3)) / denominator
    return {"mse": reduce(error), "nll": reduce(nll), "variance": reduce(variance),
            "standardized_error": reduce(error / variance)}


@torch.no_grad()
def evaluate(model, names, cache, mean, std, output):
    model.eval()
    records = []
    # Fixed bins in standardized-feature variance units. Fractional content
    # tokens, rather than encoder padding/display borders, provide the mass.
    log_edges = torch.linspace(-6., 4., 21, device="cuda")
    local_mass = torch.zeros(20, dtype=torch.float64, device="cuda")
    local_variance = torch.zeros_like(local_mass)
    local_error = torch.zeros_like(local_mass)
    for name in names:
        item = torch.load(cache / "development" / (name + ".pt"), weights_only=True)
        full = torch.load(cache / "targets" / (name + ".pt"), weights_only=True)
        partial = (item["features"].cuda().float() - mean) / std
        target = (full["features"][None].cuda().float() - mean) / std
        weight = full["weight"][None].cuda().expand(len(partial), -1, -1, -1)
        predicted, logvar = model(partial, item["reveal"].cuda(), weight)
        token_variance = logvar.exp().mean(1)
        token_error = (predicted - target).square().mean(1)
        token_weight = weight[:, 0]
        bin_index = torch.bucketize(token_variance.log().contiguous(), log_edges[1:-1]).flatten()
        for accumulator, values in ((local_mass, token_weight),
                                    (local_variance, token_weight * token_variance),
                                    (local_error, token_weight * token_error)):
            accumulator += torch.bincount(bin_index, weights=values.flatten().double(), minlength=20)
        values = {"predictor": metrics(predicted, logvar, target, weight),
                  "partial_feature": metrics(partial, torch.zeros_like(partial), target, weight),
                  "train_mean": metrics(torch.zeros_like(partial), torch.zeros_like(partial), target, weight)}
        for i, row in enumerate(item["records"]):
            records.append({**{k: v for k, v in row.items() if k != "xy"},
                            **{key + "_" + stat: arr[i].item()
                               for key, stats in values.items() for stat, arr in stats.items()}})
    metric_keys = [k for k in records[0] if k.startswith(("predictor_", "partial_feature_", "train_mean_"))]
    grouped = defaultdict(list)
    by_image = defaultdict(list)
    for row in records:
        grouped[(row["kind"], row["budget"])].append(row)
        by_image[row["image"]].append(row)
    image_metrics = [{"image": name, **{k: float(np.mean([r[k] for r in rows])) for k in metric_keys}}
                     for name, rows in by_image.items()]
    overall = {k: float(np.mean([r[k] for r in image_metrics])) for k in metric_keys}
    by_budget = [{"kind": kind, "budget": budget, "images": len(rows),
                  "short_histories": sum(r["actual_budget"] < budget for r in rows),
                  **{k: float(np.mean([r[k] for r in rows])) for k in metric_keys}}
                 for (kind, budget), rows in grouped.items()]
    # Calibration is at image/state scale: bins are not independent pixels or trials.
    ordered = sorted(records, key=lambda r: r["predictor_variance"])
    calibration = [{"bin": i, "image_states": len(chunk),
                    "mean_variance": float(np.mean([r["predictor_variance"] for r in chunk])),
                    "mean_squared_error": float(np.mean([r["predictor_mse"] for r in chunk]))}
                   for i, chunk in enumerate(np.array_split(np.array(ordered, dtype=object), 10))]
    paired = []
    for kind in ("human", "random"):
        rows = {(r["image"], r["budget"]): r for r in records if r["kind"] == kind}
        for low, high in ((1, 4), (4, 8), (1, 8)):
            diffs = [rows[(name, low)]["predictor_mse"] - rows[(name, high)]["predictor_mse"] for name in names]
            paired.append({"kind": kind, "from_budget": low, "to_budget": high,
                           "mean_mse_reduction": float(np.mean(diffs)),
                           "images_improved_fraction": float(np.mean(np.asarray(diffs) > 0))})
    local_calibration = []
    edges = log_edges.exp().cpu().tolist()
    for i, (mass, variance_sum, error_sum) in enumerate(zip(local_mass.cpu().tolist(),
                                                          local_variance.cpu().tolist(),
                                                          local_error.cpu().tolist())):
        local_calibration.append({"bin": i, "variance_lower": edges[i], "variance_upper": edges[i + 1],
                                  "content_token_weight": mass,
                                  "mean_variance": variance_sum / mass if mass else None,
                                  "mean_squared_error": error_sum / mass if mass else None})
    summary = {"overall_image_macro": overall, "by_kind_budget": by_budget,
               "calibration_image_state_deciles": calibration, "paired_same_history": paired,
               "calibration_local_tokens_fixed_bins": local_calibration,
               "local_calibration_definition": "Channel-mean variance vs channel-mean squared error; 20 fixed log-variance bins from -6 to 4, weighted by token content-pixel fraction; descriptive across all development states",
               "images": len(names), "states": len(records),
               "baseline_nll": "Both baselines use fixed unit variance in train-standardized feature units"}
    write_json(output, {"summary": summary, "images": image_metrics, "states": records})
    return summary


def main(args):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    run, root = Path(args.run), Path(args.root)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    # Parse split metadata, but never open holdout image files or produce holdout states.
    entries = {name: info for name, info in manifest["images"].items() if info["split"] in ("train", "development")}
    train_names = sorted(n for n, v in entries.items() if v["split"] == "train")
    dev_names = sorted(n for n, v in entries.items() if v["split"] == "development")
    assert len(train_names) == 3343 and len(dev_names) == 371
    global train_names_set
    train_names_set = set(train_names)
    histories = defaultdict(list)
    for row in json.loads(Path(manifest["raw_fixations"]).read_text()):
        if row["name"] in entries:
            histories[row["name"]].append(row)
    config = {"seed": SEED, "epochs": args.epochs, "batch_size": args.batch_size,
              "revision": args.revision, "model": "dinov2_vits14", "train_images": 3343,
              "development_images": 371, "holdout_images_used": 0,
              "manifest_sha256": digest(manifest_path), "fixations_sha256": digest(manifest["raw_fixations"]),
              "observer_sha256": digest(Path(__file__).with_name("observer.py")),
              "train_code_sha256": digest(__file__), "learning_rate": .0003,
              "target_features": "x_norm_patchtokens, channel train z-score, content fraction weighted",
              "variance": "diagonal Gaussian per feature; log variance clamped [-6,4]",
              "rgb_normalization": "ImageNet mean/std; AMP float16 frozen encoder, float32 head",
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(), "state": "preparing"}
    write_json(run / "config.json", config)
    encoder = Encoder(root / "vendor" / "dinov2")
    weights = Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints" / "dinov2_vits14_pretrain.pth"
    config["weights_sha256"] = digest(weights)
    config["weights_url"] = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
    # Cache identity includes all code/config inputs affecting target or partial states.
    cache_identity = {k: v for k, v in config.items() if k not in ("gpu", "state")}
    cache_key = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()[:20]
    cache = root / "features" / cache_key
    cache.mkdir(parents=True, exist_ok=True)
    config["cache"] = str(cache)
    write_json(run / "config.json", config)
    norm = prepare_cache(entries, train_names, dev_names, histories, encoder, cache, run)
    save(run / "normalization.pt", norm)
    mean = norm["mean"].cuda()[None, :, None, None]
    std = norm["std"].cuda()[None, :, None, None]
    model = Predictor().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    first_epoch, first_batch, best = 0, 0, math.inf
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        assert resume["cache_key"] == cache_key, "Resume input/config mismatch"
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        first_epoch, first_batch, best = resume["epoch"], resume["batch"], resume["best"]
        prior_best = Path(args.resume).parent / "best.pt"
        if prior_best.exists():
            saved_best = torch.load(prior_best, map_location="cpu", weights_only=False)
            assert saved_best["dev_nll"] == best, "Resume best checkpoint mismatch"
            shutil.copyfile(prior_best, run / "best.pt")
        else:
            assert not math.isfinite(best), "Missing best checkpoint for resumed run"
    def checkpoint(epoch, batch):
        save(run / "last.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                              "epoch": epoch, "batch": batch, "best": best, "cache_key": cache_key,
                              "normalization": norm, "config": config})
    start = time.monotonic()
    config["state"] = "training"
    write_json(run / "config.json", config)
    for epoch in range(first_epoch, args.epochs):
        shuffled = list(np.array(train_names)[rng_for("order", epoch).permutation(len(train_names))])
        dataset = TrainingStates(shuffled, entries, histories, cache / "targets", epoch)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True)
        model.train()
        train_loss, examples = 0., 0
        state_counts = defaultdict(int)
        for batch_index, batch in enumerate(loader):
            if epoch == first_epoch and batch_index < first_batch:
                continue
            partial = (encoder(batch["rgb"]) - mean) / std
            target = (batch["target"].cuda() - mean) / std
            weight, reveal = batch["weight"].cuda(), batch["reveal"].cuda()
            predicted, logvar = model(partial, reveal, weight)
            loss = metrics(predicted, logvar, target, weight)["nll"].mean()
            assert torch.isfinite(loss), "Nonfinite training objective"
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            train_loss += loss.item() * len(partial)
            examples += len(partial)
            for kind, requested, actual in zip(batch["kind"], batch["budget"], batch["actual_budget"]):
                state_counts[f"{kind}:requested={requested.item()}:actual={actual.item()}"] += 1
            if (batch_index + 1) % 100 == 0 or STOP:
                checkpoint(epoch, batch_index + 1)
                print(json.dumps({"stage": "train", "epoch": epoch + 1, "batch": batch_index + 1,
                                  "nll": train_loss / examples, "seconds": time.monotonic() - start}), flush=True)
            if STOP:
                config["state"] = "interrupted_resumable"
                write_json(run / "config.json", config)
                return
        summary = evaluate(model, dev_names, cache, mean, std, run / f"development_epoch_{epoch + 1:02d}.json")
        dev_nll = summary["overall_image_macro"]["predictor_nll"]
        if dev_nll < best:
            best = dev_nll
            save(run / "best.pt", {"model": model.state_dict(), "normalization": norm,
                                   "epoch": epoch + 1, "dev_nll": best, "config": config})
        checkpoint(epoch + 1, 0)
        report = {"epoch": epoch + 1, "train_nll": train_loss / examples if examples else None,
                  "train_images_this_process": examples, "best_dev_nll": best, "development": summary,
                  "training_state_counts_this_process": dict(state_counts),
                  "seconds": time.monotonic() - start}
        with open(run / "epochs.jsonl", "a") as stream:
            stream.write(json.dumps(report, allow_nan=False) + "\n")
        print(json.dumps({"stage": "epoch_complete", **report}), flush=True)
        if STOP:
            config["state"] = "interrupted_resumable"
            write_json(run / "config.json", config)
            return
    config.update(state="completed", seconds_training=time.monotonic() - start,
                  best_development_nll=best, cuda_peak_memory_bytes=torch.cuda.max_memory_allocated())
    write_json(run / "config.json", config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--manifest", default="/mnt/disk2/youyouyang/proposal2/coco_freeview/manifest.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume")
    args = parser.parse_args()
    try:
        main(args)
    except BaseException as error:
        write_json(Path(args.run) / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
