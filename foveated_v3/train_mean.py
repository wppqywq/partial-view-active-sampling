"""Direct MSE mean predictor for continuous foveation; numerical work in Slurm only.

Reuses the frozen DINO encoder and complete-image target features. Every partial
input is produced by the new observer: fixed development features are cached in
a new provenance namespace, and training features are generated online.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time

if not os.environ.get("SLURM_JOB_ID"):
    raise RuntimeError("Run this program inside the supplied Slurm job")

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import frozen_data as P
import frozen_mean as M
from frozen_legacy_observer import PartialObserver as LegacyObserver
from observer import FoveatedObserver

SEED = 20261005
ROOT = Path("/mnt/disk2/youyouyang/proposal2/foveated_v3")
MANIFEST = Path("/mnt/disk2/youyouyang/proposal2/coco_freeview/manifest.json")
MANIFEST_HASH = "7cf0f53990dad5b74657250d45197d832b5800a264c8792dc4ec3332c9541eb4"
FIXATIONS_HASH = "771167f4b5de5b9b58fb07c40d6e285d38b3b2ab9e4c6267df8a454a909da730"
MEAN_SOURCE_HASH = "bf3553ddec2fd554b4f33a8f85a71991728a358d3ed58a9dff89a9f6e54388ab"
TARGETS = M.UPSTREAM / "features/73d7ae38f654227711f1/targets"
STOP = False


def request_stop(*_):
    global STOP
    STOP = True


signal.signal(signal.SIGUSR1, request_stop)
signal.signal(signal.SIGTERM, request_stop)


def rng_for(*parts):
    value = ":".join(map(str, (SEED,) + parts)).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(value).digest()[:8], "big"))


def observer(entry, options):
    assert entry["split"] in ("train", "development")
    with Image.open(entry["path"]) as image:
        return FoveatedObserver(image, content_box=entry["display_content_box"], **options)


def history_for(obs, rows, rng, kind, budget):
    """Select subject before its length is considered; never invent human points."""
    row = rows[int(rng.integers(len(rows)))]
    assert len(row["X"]) == len(row["Y"]) > 0
    if kind == "human":
        points = np.column_stack((row["X"], row["Y"]))[:budget].tolist()
    else:
        points = [[float(row["X"][0]), float(row["Y"][0])]]
        xy = obs.choice_grid()["xy"]
        initial_view = obs.observe(points)
        resolution, valid = initial_view["resolution_map"], initial_view["valid_mask"]
        for index in rng.permutation(len(xy)):
            if len(points) >= budget:
                break
            proposal = xy[index].tolist()
            increase = obs.resolution_increase(proposal, resolution)
            if increase[valid].mean() > 1e-8:
                points.append(proposal)
                resolution = resolution + increase
        assert len(points) == budget, "Random history exhausted usable candidates"
    return points, {"subject": row["subject"], "actual_budget": len(points)}


class TrainingStates(Dataset):
    def __init__(self, names, entries, histories, options, maximum_budget, epoch):
        self.names, self.entries, self.histories = names, entries, histories
        self.options, self.maximum_budget, self.epoch = options, maximum_budget, epoch

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        rng = rng_for("train", self.epoch, name)
        kind = "human" if rng.random() < .5 else "random"
        obs = observer(self.entries[name], self.options)
        # Draw a trajectory first and a supported prefix second, avoiding a
        # pile-up of padded/duplicated short histories at the maximum budget.
        points, info = history_for(obs, self.histories[name], rng, kind, self.maximum_budget)
        budget = int(rng.integers(1, len(points) + 1))
        points = points[:budget]
        view = obs.observe(points)
        target = torch.load(TARGETS / (name + ".pt"), weights_only=True)
        return {"rgb": P.rgb_tensor(view["rgb"]),
                "resolution": P.pool_mask(view["resolution_map"]),
                "weight": target["weight"].float(), "target": target["features"].float(),
                "budget": budget, "actual_budget": len(points), "kind": kind}


def checked_view(obs, points, target):
    view = obs.observe(points)
    assert view["rgb"].shape == (280, 448, 3)
    assert np.isfinite(view["rgb"]).all() and np.isfinite(view["resolution_map"]).all()
    assert np.all((view["rgb"] >= 0) & (view["rgb"] <= 1))
    assert np.all((view["resolution_map"] >= 0) & (view["resolution_map"] <= 1))
    assert torch.equal(P.pool_mask(view["valid_mask"]), target["weight"])
    return view


@torch.no_grad()
def prepare_cache(entries, train_names, dev_names, histories, options, budget, encoder, cache, run, cache_id):
    """Only complete targets reused; no old partial input is ever loaded."""
    cache.mkdir(parents=True, exist_ok=True)
    index_path = cache / "index.json"
    prior = json.loads(index_path.read_text()) if index_path.exists() else None
    if prior:
        assert prior["cache_id"] == cache_id
    target_hashes = {}
    for i, name in enumerate(train_names + dev_names):
        assert P.digest(entries[name]["path"]) == entries[name]["sha256"]
        target_hashes[name] = P.digest(TARGETS / (name + ".pt"))
        if i == 0:
            obs = observer(entries[name], options)
            with Image.open(entries[name]["path"]) as image:
                legacy = LegacyObserver(image, content_box=entries[name]["display_content_box"])
            assert np.array_equal(obs.target(), legacy.target()), "Complete target preprocessing changed"
            target = torch.load(TARGETS / (name + ".pt"), weights_only=True)
            checked_view(obs, [], target)
    if prior:
        assert prior["target_sha256"] == target_hashes, "Frozen complete-target cache changed"
    state_rows, file_hashes = [], {}
    for i, name in enumerate(dev_names):
        path = cache / (name + ".pt")
        if path.exists():
            item = torch.load(path, weights_only=True)
            assert item["cache_id"] == cache_id and item["image"] == name
            if prior:
                assert P.digest(path) == prior["development_sha256"][name]
        else:
            obs = observer(entries[name], options)
            target = torch.load(TARGETS / (name + ".pt"), weights_only=True)
            images, resolutions, records = [], [], []
            for kind in ("human", "random"):
                points, info = history_for(obs, histories[name], rng_for("development", name, kind), kind, budget)
                previous = None
                for b in range(1, len(points) + 1):
                    prefix = points[:b]
                    view = checked_view(obs, prefix, target)
                    if previous is not None:
                        assert np.all(view["resolution_map"] + 1e-7 >= previous)
                    previous = view["resolution_map"]
                    images.append(P.rgb_tensor(view["rgb"]))
                    resolutions.append(P.pool_mask(view["resolution_map"]))
                    records.append({"image": name, "kind": kind, "budget": b, "actual_budget": b,
                                    "subject": info["subject"], "xy": prefix})
            # Offline replay/scoring uses one RGB image per encoder forward;
            # keep this convention here to avoid batch-dependent AMP variation.
            features = torch.cat([encoder(rgb[None]).cpu().half() for rgb in images])
            item = {"cache_id": cache_id, "image": name, "features": features,
                    "resolution": torch.stack(resolutions), "records": records}
            P.save(path, item)
        state_rows.extend(item["records"])
        file_hashes[name] = P.digest(path)
        if (i + 1) % 25 == 0 or i + 1 == len(dev_names):
            print(json.dumps({"stage": "development_cache", "images": i + 1, "total": len(dev_names)}), flush=True)
        if STOP:
            raise InterruptedError("Interrupted while preparing reusable new partial cache")
    result = {"cache_id": cache_id, "target_sha256": target_hashes, "development_sha256": file_hashes,
              "states": len(state_rows), "images": len(dev_names)}
    P.write_json(index_path, result)
    P.write_json(run / "cache_index.json", result)
    P.write_json(run / "development_states.json", state_rows)


@torch.no_grad()
def evaluate(model, names, cache, mean, std, path):
    model.eval()
    records, image_rows = [], []
    keys = ("direct_mse", "raw_feature_mse", "zero_mse")
    for name in names:
        item = torch.load(cache / (name + ".pt"), weights_only=True)
        target = torch.load(TARGETS / (name + ".pt"), weights_only=True)
        truth = (target["features"][None].cuda().float() - mean) / std
        weight = target["weight"][None].cuda()
        current = []
        for j in range(0, len(item["records"]), 8):
            features = (item["features"][j:j + 8].cuda().float() - mean) / std
            resolution = item["resolution"][j:j + 8].cuda()
            weights = weight.expand(len(features), -1, -1, -1)
            predicted = model(features, resolution, weights)
            values = {"direct_mse": M.mse_per_image(predicted, truth, weights),
                      "raw_feature_mse": M.mse_per_image(features, truth, weights),
                      "zero_mse": M.mse_per_image(torch.zeros_like(features), truth, weights)}
            assert all(torch.isfinite(v).all() for v in values.values())
            for k, row in enumerate(item["records"][j:j + 8]):
                current.append({**{key: val for key, val in row.items() if key != "xy"},
                                **{key: val[k].item() for key, val in values.items()}})
        # Equal kind weight, regardless of the selected human's trajectory length.
        image_rows.append({"image": name, **{key: float(np.mean([
            np.mean([r[key] for r in current if r["kind"] == kind])
            for kind in ("human", "random")])) for key in keys}})
        records.extend(current)
    overall = {key: float(np.mean([row[key] for row in image_rows])) for key in keys}
    strata, paired = [], []
    budgets = sorted({r["budget"] for r in records})
    for kind in ("human", "random"):
        for budget in budgets:
            selected = [r for r in records if r["kind"] == kind and r["budget"] == budget]
            if selected:
                strata.append({"kind": kind, "budget": budget, "images": len(selected),
                               **{key: float(np.mean([r[key] for r in selected])) for key in keys}})
        lookup = {(r["image"], r["budget"]): r for r in records if r["kind"] == kind}
        for budget in budgets[1:]:
            eligible = [name for name in names if (name, budget) in lookup]
            gains = [lookup[name, budget - 1]["direct_mse"] - lookup[name, budget]["direct_mse"]
                     for name in eligible]
            if gains:
                paired.append({"kind": kind, "from_budget": budget - 1, "to_budget": budget,
                               "images": len(gains), "mean_mse_reduction": float(np.mean(gains)),
                               "images_improved_fraction": float(np.mean(np.array(gains) > 0))})
    summary = {"images": len(names), "states": len(records), "overall_image_macro": overall,
               "aggregation": "mean available budgets within kind, equal human/random kinds, equal images",
               "by_kind_budget": strata, "paired_same_history": paired,
               "units": "train-standardized complete DINO feature content-weighted MSE"}
    P.write_json(path, {"summary": summary, "images": image_rows, "states": records})
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
    source = Path(__file__).parent
    assert P.digest(source / "frozen_data.py") == M.DATA_SOURCE_HASH
    assert P.digest(source / "frozen_mean.py") == MEAN_SOURCE_HASH
    assert P.digest(source / "frozen_legacy_observer.py") == M.OBSERVER_HASH
    assert P.digest(M.V1 / "normalization.pt") == M.NORMALIZATION_HASH
    assert P.digest(MANIFEST) == MANIFEST_HASH
    manifest = json.loads(MANIFEST.read_text())
    assert P.digest(manifest["raw_fixations"]) == FIXATIONS_HASH
    entries = {name: item for name, item in manifest["images"].items() if item["split"] in ("train", "development")}
    train_names = sorted(name for name, item in entries.items() if item["split"] == "train")
    dev_names = sorted(name for name, item in entries.items() if item["split"] == "development")
    assert (len(train_names), len(dev_names)) == (3343, 371)
    histories = defaultdict(list)
    for row in json.loads(Path(manifest["raw_fixations"]).read_text()):
        if row["name"] in entries:
            histories[row["name"]].append(row)
    assert all(histories[name] for name in entries)
    lengths = [len(row["X"]) for name in train_names for row in histories[name]]
    maximum_budget = int(math.ceil(float(np.mean(lengths))))
    budget_info = json.loads(Path(args.budget_json).read_text())
    assert budget_info["maximum_budget"] == maximum_budget
    assert budget_info["manifest_sha256"] == MANIFEST_HASH
    assert budget_info["raw_fixations_sha256"] == FIXATIONS_HASH
    assert budget_info["budget_rule"] == "ceil(training_trial_mean_fixation_count)"
    assert maximum_budget >= 8
    protocol = json.loads((source / "protocol.json").read_text())
    assert protocol == budget_info, "Use the exact preparation protocol as the budget artifact"
    assert protocol["source_sha256"]["observer.py"] == P.digest(source / "observer.py")
    assert protocol["implementation_checks"].startswith("passed;")
    visual_review = json.loads(Path(args.visual_review_json).read_text())
    assert visual_review["state"] == "accepted"
    assert isinstance(visual_review["reviewer"], str) and visual_review["reviewer"].strip()
    assert visual_review["protocol_sha256"] == P.digest(source / "protocol.json")
    assert visual_review["observer_sha256"] == P.digest(source / "observer.py")
    options = protocol["observer_kwargs"]
    norm = torch.load(M.V1 / "normalization.pt", weights_only=True)
    assert norm["train_names"] == train_names
    shutil.copyfile(M.V1 / "normalization.pt", run / "normalization.pt")
    source_hashes = {p.name: P.digest(p) for p in sorted(source.glob("*.py"))}
    spec = {"seed": SEED, "model": "DirectMean", "initialization": "from scratch; zero final mean",
            "body_channels": 128, "residual": False, "variance_head": False,
            "state_channel": "content-only continuous relative spatial resolution, average pooled14",
            "observer": options, "source_sha256": source_hashes,
            "protocol_sha256": P.digest(source / "protocol.json"), "budget_sha256": P.digest(args.budget_json),
            "visual_review_sha256": P.digest(args.visual_review_json),
            "manifest_sha256": MANIFEST_HASH, "fixations_sha256": FIXATIONS_HASH,
            "normalization_sha256": M.NORMALIZATION_HASH, "target_cache": str(TARGETS),
            "encoder_revision": M.REVISION, "encoder_weights_sha256": M.WEIGHTS_HASH,
            "train_images": len(train_names), "development_images": len(dev_names), "holdout_images_used": 0,
            "maximum_budget": maximum_budget, "train_trajectory_mean": float(np.mean(lengths)),
            "train_trajectory_count": len(lengths), "budget_rounding": "ceil(train-only trial mean)",
            "maximum_epochs": 80, "minimum_epochs": 10, "batch_size": 8,
            "optimizer": {"name": "AdamW", "lr": .001, "weight_decay": .0001, "gradient_clip_norm": 1.},
            "scheduler": {"name": "ReduceLROnPlateau", "factor": .5, "patience": 3, "min_lr": 1e-5,
                          "threshold": 1e-4, "threshold_mode": "rel"},
            "early_stop": {"patience": 10, "relative_improvement": 1e-4},
            "training_states": "online new observer only; 50%human/50%random; uniform available prefix budget",
            "development_states": "fixed subject independent of length, every real prefix; no human padding",
            "encoder_batch_size": {"training": 8, "development_cache": 1, "partial_cache_dtype": "float16"},
            "development_mean_head_batch_size": 8,
            "torch": str(torch.__version__), "cuda": torch.version.cuda}
    experiment_id = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    cache_id = hashlib.sha256(json.dumps({"observer": options, "seed": SEED, "budget": maximum_budget,
        "manifest": MANIFEST_HASH, "fixations": FIXATIONS_HASH, "sources": source_hashes}, sort_keys=True).encode()).hexdigest()
    cache = ROOT / "partial_features" / cache_id
    config = {**spec, "experiment_id": experiment_id, "cache_id": cache_id, "partial_cache": str(cache),
              "development_cache": str(cache), "observer_sha256": P.digest(source / "observer.py"),
              "state": "preparing_cache", "gpu": torch.cuda.get_device_name()}
    P.write_json(run / "config.json", config)
    encoder = M.load_frozen()
    prepare_cache(entries, train_names, dev_names, histories, options, maximum_budget, encoder, cache, run, cache_id)
    torch.manual_seed(SEED)
    model = M.DirectMean().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=3,
        threshold=1e-4, threshold_mode="rel", min_lr=1e-5)
    first_epoch = first_batch = bad_epochs = 0
    best = significant_best = float("inf")
    best_payload = best_summary = None
    history, acc = [], M.accumulator()
    if args.resume:
        loaded = torch.load(args.resume, map_location="cpu", weights_only=False)
        assert loaded["experiment_id"] == experiment_id, "Resume input/source/protocol mismatch"
        model.load_state_dict(loaded["model"])
        optimizer.load_state_dict(loaded["optimizer"])
        scheduler.load_state_dict(loaded["scheduler"])
        first_epoch, first_batch, bad_epochs = loaded["epoch"], loaded["batch"], loaded["bad_epochs"]
        best, significant_best = loaded["best"], loaded["significant_best"]
        best_payload, best_summary = loaded["best_snapshot"], loaded["best_metrics"]
        history, acc = loaded["history"], loaded["accumulator"]
        M.restore_rng(loaded["rng"])
        if best_payload is not None:
            P.save(run / "best.pt", best_payload)
            P.write_json(run / "best_metrics.json", best_summary)
        if loaded["stop_status"] in ("early_stopped", "budget_exhausted"):
            raise RuntimeError("Training already terminal; do not extend the fixed budget")
    assert not encoder.training and all(not p.requires_grad for p in encoder.parameters())
    mean = norm["mean"].cuda()[None, :, None, None]
    std = norm["std"].cuda()[None, :, None, None]
    with open(run / "curves.jsonl", "w") as stream:
        for row in history:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    started = time.monotonic()

    def payload(epoch, batch, status, current_acc):
        return {"model": M.cpu_copy(model.state_dict()), "optimizer": M.cpu_copy(optimizer.state_dict()),
                "scheduler": scheduler.state_dict(), "rng": M.rng_state(), "epoch": epoch, "batch": batch,
                "best": best, "significant_best": significant_best, "bad_epochs": bad_epochs,
                "accumulator": M.cpu_copy(current_acc), "normalization": norm, "experiment_id": experiment_id,
                "stop_status": status, "history": list(history), "best_metrics": best_summary, "config": dict(config)}

    def save_last(epoch, batch, status, current_acc):
        value = payload(epoch, batch, status, current_acc)
        value["best_snapshot"] = best_payload
        P.save(run / "last.pt", value)

    def progress(status, epoch):
        config.update(state=status, completed_epochs=epoch, best_dev_mse=best if np.isfinite(best) else None,
                      best_epoch=best_payload["epoch"] if best_payload is not None else None,
                      best_checkpoint_sha256=P.digest(run / "best.pt") if best_payload is not None else None,
                      bad_epochs=bad_epochs, seconds_training_process=time.monotonic() - started,
                      current_lr=max(g["lr"] for g in optimizer.param_groups))
        P.write_json(run / "config.json", config)

    progress("training", first_epoch)
    for epoch in range(first_epoch, 80):
        names = list(np.asarray(train_names)[rng_for("order", epoch).permutation(len(train_names))])
        dataset = TrainingStates(names, entries, histories, options, maximum_budget, epoch)
        loader = DataLoader(dataset, batch_size=8, num_workers=2, pin_memory=True,
                            generator=torch.Generator().manual_seed(SEED + epoch))
        model.train()
        if epoch != first_epoch:
            acc = M.accumulator()
        state_counts = defaultdict(int, acc["state_counts"])
        lr_used = max(g["lr"] for g in optimizer.param_groups)
        for batch_index, batch in enumerate(loader):
            if epoch == first_epoch and batch_index < first_batch:
                continue
            features = (encoder(batch["rgb"]) - mean) / std
            target = (batch["target"].cuda() - mean) / std
            weight, resolution = batch["weight"].cuda(), batch["resolution"].cuda()
            predicted = model(features, resolution, weight)
            errors = M.mse_per_image(predicted, target, weight)
            assert predicted.shape == target.shape and torch.isfinite(errors).all()
            optimizer.zero_grad(set_to_none=True)
            errors.mean().backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            acc["mse_sum"] += errors.detach().sum().item()
            acc["images"] += len(features)
            for kind, budget, actual in zip(batch["kind"], batch["budget"], batch["actual_budget"]):
                assert budget.item() == actual.item()
                state_counts[f"{kind}:budget={budget.item()}"] += 1
            acc["state_counts"] = dict(state_counts)
            if (batch_index + 1) % 100 == 0 or STOP:
                save_last(epoch, batch_index + 1, "training", acc)
                print(json.dumps({"stage": "train", "epoch": epoch + 1, "batch": batch_index + 1,
                    "mse": acc["mse_sum"] / acc["images"], "lr": lr_used}), flush=True)
            if STOP:
                progress("interrupted_resumable", epoch)
                return 75
        assert acc["images"] == len(train_names)
        assert {int(key.split("budget=")[1]) for key in state_counts} == set(range(1, maximum_budget + 1))
        summary = evaluate(model, dev_names, cache, mean, std, run / f"development_epoch_{epoch + 1:03d}.json")
        dev_mse = summary["overall_image_macro"]["direct_mse"]
        relative = (significant_best - dev_mse) / abs(significant_best) if np.isfinite(significant_best) else None
        if relative is None or relative >= 1e-4:
            significant_best, bad_epochs = dev_mse, 0
        else:
            bad_epochs += 1
        improved = dev_mse < best
        if improved:
            best, best_summary = dev_mse, summary
        scheduler.step(dev_mse)
        status = "early_stopped" if epoch + 1 >= 10 and bad_epochs >= 10 else "budget_exhausted" if epoch + 1 == 80 else "training"
        record = {"epoch": epoch + 1, "train_mse": acc["mse_sum"] / acc["images"], "development_mse": dev_mse,
                  "best_development_mse": best, "lr_used": lr_used, "lr_after_scheduler": max(g["lr"] for g in optimizer.param_groups),
                  "bad_epochs": bad_epochs, "stop_status": status, "training_state_counts": dict(state_counts), "development": summary}
        history.append(record)
        with open(run / "curves.jsonl", "a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if improved:
            best_payload = payload(epoch + 1, 0, status, M.accumulator())
            best_payload["dev_mse"] = best
            P.save(run / "best.pt", best_payload)
            P.write_json(run / "best_metrics.json", best_summary)
        save_last(epoch + 1, 0, status, M.accumulator())
        progress(status, epoch + 1)
        print(json.dumps({"stage": "epoch_complete", "epoch": epoch + 1, "development_mse": dev_mse,
                          "best_development_mse": best, "status": status}), flush=True)
        if status in ("early_stopped", "budget_exhausted"):
            frozen = {"stage": "mean_only_frozen", "terminal_state": status,
                      "experiment_id": experiment_id, "best_checkpoint": str(run / "best.pt"),
                      "best_checkpoint_sha256": P.digest(run / "best.pt"),
                      "best_epoch": best_payload["epoch"], "best_dev_mse": best,
                      "normalization_sha256": P.digest(run / "normalization.pt"),
                      "protocol_sha256": P.digest(source / "protocol.json"),
                      "observer_sha256": P.digest(source / "observer.py"),
                      "source_sha256": source_hashes, "maximum_budget": maximum_budget,
                      "target_cache": str(TARGETS), "development_cache": str(cache),
                      "best_metrics": best_summary,
                      "next_stage": "independent variance head; mean and all shared parameters frozen"}
            frozen["mean_identity_hash"] = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
            P.write_json(run / "frozen_mean.json", frozen)
            return 0
        if STOP:
            progress("interrupted_resumable", epoch + 1)
            return 75
    raise RuntimeError("No terminal state reached")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--budget-json", required=True)
    parser.add_argument("--visual-review-json", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    try:
        code = main(args)
    except BaseException as error:
        P.write_json(Path(args.run_dir) / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    raise SystemExit(code)
