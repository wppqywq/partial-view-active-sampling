"""Fresh expected gains for the reviewed continuous-foveation observer.

All target/after-view access is confined to label generation. Online inputs use
the before state and candidate geometry only. Execute through Slurm.
"""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

if not os.environ.get("SLURM_JOB_ID"):
    raise RuntimeError("Run fresh gain generation/training inside Slurm")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import FrozenObserver, P, T
from observer import FoveatedObserver
from signals import online_vectors, realized_labels

SEED = 20261007
STOP = False
MANIFEST = Path("/mnt/disk2/youyouyang/proposal2/coco_freeview/manifest.json")
ROOT = Path("/mnt/disk2/youyouyang/proposal2/foveated_v3")


def request_stop(*_):
    global STOP
    STOP = True


# Restore handlers after importing frozen training modules.
signal.signal(signal.SIGUSR1, request_stop)
signal.signal(signal.SIGTERM, request_stop)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def dump(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def rng_for(*parts):
    seed = hashlib.sha256(":".join(map(str, (SEED,) + parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(seed[:8], "big"))


class GainHead(nn.Module):
    """Same architecture as gain_e/gain_final: 2309 -> 256 -> 64 -> 2."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2309, 256), nn.GELU(),
                                 nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, x):
        return self.net(x)


class FrozenGain:
    """Immutable inference wrapper: before-only vectors -> signed expected gains."""
    def __init__(self, gain_run, frozen_observer):
        self.run = Path(gain_run).resolve()
        self.config = json.loads((self.run / "config.json").read_text())
        self.completion = json.loads((self.run / "completion.json").read_text())
        assert self.config["state"] in ("early_stopped", "budget_exhausted")
        assert self.completion["state"] == "completed"
        assert self.config["observer_identity"] == frozen_observer.identity
        assert self.completion["observer_identity_hash"] == frozen_observer.identity_hash
        self.best_hash = sha(self.run / "best.pt")
        self.norm_hash = sha(self.run / "normalization.pt")
        assert self.best_hash == self.completion["best_checkpoint_sha256"] == self.config["best_checkpoint_sha256"]
        assert self.norm_hash == self.completion["normalization_sha256"]
        assert self.config["cache_key"] == self.completion["cache_key"]
        for filename in ("train_gain.py", "signals.py"):
            assert sha(Path(__file__).with_name(filename)) == self.config["source_sha256"][filename]
        saved = torch.load(self.run / "best.pt", map_location="cpu", weights_only=False)
        assert saved["observer_identity_hash"] == frozen_observer.identity_hash
        assert saved["cache_key"] == self.config["cache_key"]
        norm = torch.load(self.run / "normalization.pt", weights_only=True)
        assert norm["cache_key"] == self.config["cache_key"]
        assert norm["observer_identity_hash"] == frozen_observer.identity_hash
        for key in ("x_mean", "x_std", "y_mean", "y_std"):
            assert torch.equal(norm[key], saved["normalization"][key])
        self.norm = {k: v.cuda() if torch.is_tensor(v) else v for k, v in norm.items()}
        self.model = GainHead().cuda().eval().requires_grad_(False)
        self.model.load_state_dict(saved["model"])

    @torch.no_grad()
    def __call__(self, before_vectors):
        assert before_vectors.ndim == 2 and before_vectors.shape[1] == 2309
        x = (before_vectors.cuda().float() - self.norm["x_mean"]) / self.norm["x_std"]
        value = self.model(x) * self.norm["y_std"] + self.norm["y_mean"]
        assert value.shape == (len(before_vectors), 2) and torch.isfinite(value).all()
        return value

    def assert_unchanged(self):
        assert sha(self.run / "best.pt") == self.best_hash
        assert sha(self.run / "normalization.pt") == self.norm_hash
        assert not self.model.training and all(not p.requires_grad for p in self.model.parameters())
        saved = torch.load(self.run / "best.pt", map_location="cpu", weights_only=False)["model"]
        assert all(torch.equal(value.detach().cpu(), saved[key]) for key, value in self.model.state_dict().items())


def load_gain(gain_run, frozen_observer):
    return FrozenGain(gain_run, frozen_observer)


@torch.no_grad()
def cache_image(name, entry, histories, frozen, count, cache_key):
    obs = T.observer(entry, frozen.options)
    xy = obs.choice_grid()["xy"]
    fixed_weights = torch.stack([P.pool_mask(obs.local_weights(point)) for point in xy])
    empty = obs.observe([])
    content = P.pool_mask(empty["valid_mask"])[None].cuda()
    target = frozen.target_for_labels(name)
    states, all_x, all_y, all_u = [], [], [], []
    offset = 0
    for kind in ("human", "random"):
        points, metadata = T.history_for(obs, histories, rng_for("history", name, kind),
                                         kind, frozen.maximum_budget - 1)
        for budget in range(1, len(points) + 1):
            history = points[:budget]
            view = obs.observe(history)
            grid, eligible = obs.candidates(view["resolution_map"])
            assert np.array_equal(grid, xy)
            ids = np.flatnonzero(eligible)
            chosen = rng_for("candidates", name, kind, budget).choice(
                ids, size=min(count, len(ids)), replace=False)
            chosen = np.asarray(chosen, dtype=np.int64)
            state = {"image": name, "kind": kind, "budget": budget,
                     "actual_budget": len(history), "subject": metadata["subject"],
                     "history_xy": history, "requested_candidates": count,
                     "eligible_candidates": len(ids), "candidate_ids": chosen.tolist(),
                     "candidate_xy": xy[chosen].tolist(), "offset": offset, "count": len(chosen)}
            states.append(state)
            if not len(chosen):
                continue
            resolution = P.pool_mask(view["resolution_map"])[None].cuda()
            partial, mean, logvar = frozen.current(P.rgb_tensor(view["rgb"])[None], resolution, content)
            weights = fixed_weights[chosen].cuda()
            increments = torch.stack([P.pool_mask(obs.resolution_increase(point, view["resolution_map"]))
                                      for point in xy[chosen]]).cuda()
            # Must be constructed before access to any hypothetical reveal/target error.
            vector, u = online_vectors(partial, mean, logvar, content, weights,
                                        xy[chosen], obs.content_box, resolution, increments,
                                        budget, frozen.maximum_budget)
            stored = vector.cpu().half()
            assert torch.isfinite(stored).all()
            labels = []
            for j, candidate in enumerate(chosen):
                after = obs.observe(history + [xy[candidate].tolist()])
                _, after_mean, _ = frozen.current(P.rgb_tensor(after["rgb"])[None],
                    P.pool_mask(after["resolution_map"])[None].cuda(), content)
                labels.append(realized_labels(mean, after_mean, target, content,
                                               weights[j:j + 1]).cpu())
            all_x.append(stored)
            all_y.append(torch.cat(labels))
            all_u.append(u.cpu())
            offset += len(chosen)
    assert all_x, "No valid gain candidate on image"
    result = {"x": torch.cat(all_x), "y": torch.cat(all_y), "u": torch.cat(all_u),
              "states": states, "image": name, "split": entry["split"],
              "cache_key": cache_key, "observer_identity_hash": frozen.identity_hash}
    assert result["x"].shape == (offset, 2309) and result["y"].shape == (offset, 2)
    assert torch.isfinite(result["y"]).all()
    return result


def prepare_cache(entries, histories, frozen, cache, run, cache_key):
    counts = defaultdict(int)
    hashes = {}
    started = time.monotonic()
    for index, (name, entry) in enumerate(sorted(entries.items())):
        path = cache / (name + ".pt")
        hash_path = cache / (name + ".sha256")
        if path.exists() and hash_path.exists():
            assert sha(path) == hash_path.read_text().strip(), "Changed label cache"
            item = torch.load(path, weights_only=True)
            assert item["cache_key"] == cache_key and item["observer_identity_hash"] == frozen.identity_hash
            assert item["image"] == name and item["split"] == entry["split"]
        else:
            # A job may die between atomically saving a data file and recording
            # its hash. Such an orphan is recomputed, never trusted as reusable.
            assert sha(entry["path"]) == entry["sha256"]
            item = cache_image(name, entry, histories[name], frozen,
                               8 if entry["split"] == "train" else 16, cache_key)
            save(path, item)
            hash_path.write_text(sha(path) + "\n")
        hashes[name] = hash_path.read_text().strip()
        counts[entry["split"] + "_images"] += 1
        counts[entry["split"] + "_states"] += len(item["states"])
        counts[entry["split"] + "_candidates"] += len(item["x"])
        counts["empty_candidate_states"] += sum(s["count"] == 0 for s in item["states"])
        assert all(s["actual_budget"] == s["budget"] for s in item["states"])
        if (index + 1) % 10 == 0 or STOP or index + 1 == len(entries):
            progress = {"stage": "fresh_labels", "images": index + 1, "total": len(entries),
                        "elapsed_seconds": time.monotonic() - started, "counts": dict(counts),
                        "cache_key": cache_key}
            dump(run / "cache_progress.json", progress)
            print(json.dumps(progress), flush=True)
        if STOP:
            return False
    frozen.assert_unchanged()
    dump(cache / "index.json", {"cache_key": cache_key, "file_sha256": hashes, "counts": dict(counts)})
    dump(run / "cache_summary.json", {"cache_key": cache_key, "counts": dict(counts),
                                      "index_sha256": sha(cache / "index.json")})
    return True


def load_split(names, cache, training=False):
    xs, ys, us, states = [], [], [], []
    offset = 0
    for name in names:
        path = cache / (name + ".pt")
        assert sha(path) == (cache / (name + ".sha256")).read_text().strip()
        item = torch.load(path, weights_only=True)
        xs.append(item["x"])
        ys.append(item["y"])
        if not training:
            us.append(item["u"])
            states.extend([{**s, "offset": s["offset"] + offset} for s in item["states"]])
        offset += len(item["x"])
    return {"x": torch.cat(xs), "y": torch.cat(ys),
            "u": torch.cat(us) if us else None, "states": states}


def train_normalization(data):
    sums = torch.zeros(2309, dtype=torch.float64)
    squares = sums.clone()
    for chunk in data["x"].split(4096):
        value = chunk.double()
        sums += value.sum(0)
        squares += value.square().sum(0)
    mean = sums / len(data["x"])
    std = (squares / len(data["x"]) - mean.square()).clamp_min(1e-8).sqrt()
    return {"x_mean": mean.float(), "x_std": std.float(),
            "y_mean": data["y"].mean(0), "y_std": data["y"].std(0, unbiased=False).clamp_min(1e-6),
            "source": "training candidates only", "count": len(data["x"])}


def spearman(a, b):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    def ranks(x):
        order = np.argsort(x, kind="stable")
        values = x[order]
        starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
        ends = np.r_[starts[1:], len(x)]
        result = np.empty(len(x), dtype=float)
        for start, end in zip(starts, ends):
            result[order[start:end]] = (start + end - 1) / 2
        return result
    x, y = ranks(a), ranks(b)
    x, y = x - x.mean(), y - y.mean()
    return float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y)))


@torch.no_grad()
def evaluate(model, data, norm, output=None):
    model.eval()
    predictions = []
    for chunk in data["x"].split(2048):
        x = (chunk.cuda().float() - norm["x_mean"]) / norm["x_std"]
        predictions.append((model(x) * norm["y_std"] + norm["y_mean"]).cpu())
    prediction = torch.cat(predictions).numpy()
    truth, u = data["y"].numpy(), data["u"].numpy()
    assert np.isfinite(prediction).all()
    constant, scale = norm["y_mean"].cpu().numpy(), norm["y_std"].cpu().numpy()
    rows = []
    for state in data["states"]:
        if not state["count"]:
            continue
        sl = slice(state["offset"], state["offset"] + state["count"])
        p, y = prediction[sl], truth[sl]
        row = {**state, "standardized_mse": float(np.mean(((p - y) / scale) ** 2))}
        for column, label in enumerate(("global", "local")):
            picked = int(np.argmax(p[:, column]))
            row.update({label + "_mse": float(np.mean((p[:, column] - y[:, column]) ** 2)),
                        label + "_constant_mse": float(np.mean((constant[column] - y[:, column]) ** 2)),
                        label + "_spearman": spearman(p[:, column], y[:, column]),
                        label + "_u_spearman": spearman(u[sl], y[:, column]),
                        label + "_mean_realized": float(y[:, column].mean()),
                        label + "_negative_fraction": float(np.mean(y[:, column] < 0)),
                        label + "_selected_minus_uniform": float(y[picked, column] - y[:, column].mean()),
                        label + "_oracle_minus_uniform": float(y[:, column].max() - y[:, column].mean())})
        if output is not None:
            row.update(predictions=p.tolist(), realized=y.tolist(), uncertainty=u[sl].tolist())
        rows.append(row)
    metric_keys = ["standardized_mse"] + [key for key in rows[0] if key.startswith(("global_", "local_"))]
    def summarize(selected):
        images = defaultdict(list)
        for row in selected:
            images[row["image"]].append(row)
        result = {"images": len(images), "states": len(selected), "candidates": sum(s["count"] for s in selected)}
        for key in metric_keys:
            means = [float(np.mean([s[key] for s in group if s[key] is not None]))
                     for group in images.values() if any(s[key] is not None for s in group)]
            result[key] = float(np.mean(means)) if means else None
            if "spearman" in key:
                result[key + "_valid_states"] = sum(s[key] is not None for s in selected)
        return result
    strata = [{"kind": kind, "budget": b,
               **summarize([s for s in rows if s["kind"] == kind and s["budget"] == b])}
              for kind in ("human", "random") for b in sorted({s["budget"] for s in rows})]
    summary = {"image_macro": summarize(rows), "by_kind_budget": strata,
               "aggregation": "equal candidate within state, equal available states within image, equal images",
               "selection": "image-macro MSE averaged over train-standardized global and local outputs",
               "rank_validity": "Within-state Spearman; at least3 candidates; constant ranks omitted with counts",
               "interpretation": "Sampled-candidate ranking only; oracle is hindsight, not a realizable policy"}
    if output is not None:
        dump(output, {"summary": summary, "states": rows})
    return summary


def main(args):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    run = Path(args.run_dir)
    frozen = FrozenObserver(args.mean_run, args.variance_run, require_calibration=True)
    # The common loader verifies the terminal new mean, independent variance,
    # approved calibration and all immutable checkpoint/protocol bindings.
    protocol = frozen.protocol
    assert protocol["version"] == "continuous_foveation_v3"
    assert sha(MANIFEST) == protocol["manifest_sha256"]
    manifest = json.loads(MANIFEST.read_text())
    assert sha(manifest["raw_fixations"]) == protocol["raw_fixations_sha256"]
    entries = {n: e for n, e in manifest["images"].items() if e["split"] in ("train", "development")}
    train_names = sorted(n for n, e in entries.items() if e["split"] == "train")
    dev_names = sorted(n for n, e in entries.items() if e["split"] == "development")
    assert (len(train_names), len(dev_names)) == (3343, 371)
    histories = defaultdict(list)
    for row in json.loads(Path(manifest["raw_fixations"]).read_text()):
        if row["name"] in entries:
            histories[row["name"]].append(row)
    sources = {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    spec = {"version": "continuous_foveation_gain_v3", "seed": SEED,
            "observer_identity": frozen.identity, "observer_identity_hash": frozen.identity_hash,
            "source_sha256": sources, "protocol_sha256": sha(Path(args.mean_run) / "protocol.json"),
            "maximum_budget": frozen.maximum_budget, "before_budgets": list(range(1, frozen.maximum_budget)),
            "human_histories": "real prefixes only; never pad short trajectories; retain revisits",
            "history_kinds": ["human", "random"], "train_candidates_per_state": 8,
            "development_candidates_per_state": 16, "train_images": 3343, "development_images": 371,
            "holdout_images": 0, "canonical_forward_batch_size": 1,
            "architecture": [2309, 256, 64, 2], "outputs": ["global_gain", "local_gain"],
            "storage": "float16 before-only inputs; float32 signed gains, including negative labels",
            "maximum_epochs": 100, "minimum_epochs": 10, "early_stopping_patience": 10,
            "relative_improvement": 1e-4, "batch_size": 512, "learning_rate": .001,
            "weight_decay": .0001, "scheduler": "ReduceLROnPlateau factor.5 patience3 min1e-5",
            "torch": str(torch.__version__), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()}
    cache_key = identity(spec)
    cache = ROOT / "gain" / "candidates" / cache_key
    cache.mkdir(parents=True, exist_ok=True)
    identity_path = cache / "identity.json"
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == spec
    else:
        assert not any(cache.iterdir()), "Unversioned files in fresh label cache"
        dump(identity_path, spec)
    config = {**spec, "cache_key": cache_key, "cache": str(cache), "state": "preparing_fresh_labels"}
    dump(run / "config.json", config)
    dump(run / "observer_identity.json", frozen.identity)
    if not prepare_cache(entries, histories, frozen, cache, run, cache_key):
        config["state"] = "interrupted_resumable_cache"
        dump(run / "config.json", config)
        return 75
    # Loader verification is repeated at completion; no optimizer owns its params.
    del frozen
    torch.cuda.empty_cache()
    train = load_split(train_names, cache, training=True)
    dev = load_split(dev_names, cache)
    norm_cpu = train_normalization(train)
    norm_cpu.update(observer_identity_hash=spec["observer_identity_hash"], cache_key=cache_key)
    save(run / "normalization.pt", norm_cpu)
    norm = {k: v.cuda() if torch.is_tensor(v) else v for k, v in norm_cpu.items()}
    model = GainHead().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5,
        patience=3, threshold=1e-4, threshold_mode="rel", min_lr=1e-5)
    first_epoch = first_batch = bad_epochs = 0
    best = significant_best = math.inf
    best_snapshot = None
    history = []
    running_loss, running_count = 0., 0
    if args.resume:
        prior = torch.load(args.resume, map_location="cpu", weights_only=False)
        assert prior["cache_key"] == cache_key
        assert prior["state"] not in ("early_stopped", "budget_exhausted")
        model.load_state_dict(prior["model"])
        optimizer.load_state_dict(prior["optimizer"])
        scheduler.load_state_dict(prior["scheduler"])
        first_epoch, first_batch = prior["epoch"], prior["batch"]
        best, significant_best, bad_epochs = prior["best"], prior["significant_best"], prior["bad_epochs"]
        best_snapshot, history = prior["best_snapshot"], prior["history"]
        running_loss, running_count = prior["running_loss"], prior["running_count"]
        torch.set_rng_state(prior["torch_rng"])
        torch.cuda.set_rng_state_all(prior["cuda_rng"])
        if best_snapshot is not None:
            save(run / "best.pt", best_snapshot)
    def snapshot(epoch, batch, state):
        return {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "optimizer": T.M.cpu_copy(optimizer.state_dict()), "scheduler": copy.deepcopy(scheduler.state_dict()),
                "epoch": epoch, "batch": batch, "state": state, "best": best,
                "significant_best": significant_best, "bad_epochs": bad_epochs,
                "cache_key": cache_key, "observer_identity": spec["observer_identity"],
                "observer_identity_hash": spec["observer_identity_hash"], "normalization": norm_cpu,
                "config": dict(config), "history": list(history), "running_loss": running_loss,
                "running_count": running_count, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()}
    def checkpoint(epoch, batch, state):
        saved = snapshot(epoch, batch, state)
        saved["best_snapshot"] = best_snapshot
        save(run / "last.pt", saved)
    config["state"] = "training_gain"
    dump(run / "config.json", config)
    for epoch in range(first_epoch, 100):
        order = rng_for("training_order", epoch).permutation(len(train["x"]))
        if epoch != first_epoch:
            running_loss, running_count = 0., 0
        model.train()
        for batch, begin in enumerate(range(0, len(order), 512)):
            if epoch == first_epoch and batch < first_batch:
                continue
            ids = torch.from_numpy(order[begin:begin + 512])
            x = (train["x"][ids].cuda().float() - norm["x_mean"]) / norm["x_std"]
            y = (train["y"][ids].cuda() - norm["y_mean"]) / norm["y_std"]
            loss = F.mse_loss(model(x), y)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            running_loss += loss.item() * len(ids)
            running_count += len(ids)
            if STOP or (batch + 1) % 250 == 0:
                checkpoint(epoch, batch + 1, "training_gain")
            if STOP:
                config.update(state="interrupted_resumable_training", epoch=epoch, batch=batch + 1)
                dump(run / "config.json", config)
                return 75
        assert running_count == len(train["x"])
        summary = evaluate(model, dev, norm)
        score = summary["image_macro"]["standardized_mse"]
        improved = score < best
        if not math.isfinite(significant_best) or score < significant_best * (1 - 1e-4):
            significant_best, bad_epochs = score, 0
        else:
            bad_epochs += 1
        best = min(best, score)
        scheduler.step(score)
        state = "early_stopped" if epoch + 1 >= 10 and bad_epochs >= 10 else "budget_exhausted" if epoch + 1 == 100 else "training_gain"
        history.append({"epoch": epoch + 1, "training_standardized_mse": running_loss / running_count,
                        "development": summary, "lr": max(g["lr"] for g in optimizer.param_groups),
                        "bad_epochs": bad_epochs, "state": state})
        running_loss, running_count = 0., 0
        if improved:
            best_snapshot = snapshot(epoch + 1, 0, state)
            best_snapshot["development"] = summary
            save(run / "best.pt", best_snapshot)
            dump(run / "best_metrics.json", summary)
        checkpoint(epoch + 1, 0, state)
        dump(run / "curves.json", history)
        config.update(state=state, completed_epochs=epoch + 1, best_development_mse=best,
                      best_epoch=best_snapshot["epoch"], best_checkpoint_sha256=sha(run / "best.pt"))
        dump(run / "config.json", config)
        print(json.dumps({"stage": "epoch_complete", "epoch": epoch + 1,
                          "development_standardized_mse": score, "state": state}), flush=True)
        if state in ("early_stopped", "budget_exhausted"):
            model.load_state_dict(best_snapshot["model"])
            evaluate(model, dev, norm, run / "development_candidates.json")
            check = FrozenObserver(args.mean_run, args.variance_run, require_calibration=True)
            assert check.identity == spec["observer_identity"]
            check.assert_unchanged()
            dump(run / "completion.json", {"state": "completed", "best_epoch": best_snapshot["epoch"],
                "best_checkpoint_sha256": sha(run / "best.pt"), "normalization_sha256": sha(run / "normalization.pt"),
                "observer_identity_hash": spec["observer_identity_hash"], "cache_key": cache_key,
                "development_candidates_sha256": sha(run / "development_candidates.json")})
            return 0
        if STOP:
            config["state"] = "interrupted_resumable_training"
            dump(run / "config.json", config)
            return 75
    raise RuntimeError("No terminal gain status")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mean-run", required=True)
    parser.add_argument("--variance-run", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    try:
        status = main(args)
    except BaseException as error:
        dump(Path(args.run_dir) / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    raise SystemExit(status)
