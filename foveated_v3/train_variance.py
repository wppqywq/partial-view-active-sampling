"""Independent variance fit for the frozen v3 mean; no automatic calibration pass."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

if not os.environ.get("SLURM_JOB_ID"):
    raise RuntimeError("Use the supplied Slurm job")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from common import FrozenObserver, VarianceHead, P, T
from signals import spatial_average

SEED = 20261005
STOP = False


def request_stop(*_):
    global STOP
    STOP = True


signal.signal(signal.SIGUSR1, request_stop)
signal.signal(signal.SIGTERM, request_stop)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def metrics(mean, logvar, target, weight):
    error, variance = (mean - target).square(), logvar.exp()
    assert torch.isfinite(variance).all() and (variance > 0).all()
    denominator = weight.sum((1, 2, 3)) * mean.shape[1]
    def reduce(value):
        return (value * weight).sum((1, 2, 3)) / denominator
    return {"nll": reduce(.5 * (math.log(2 * math.pi) + logvar + error / variance)),
            "mse": reduce(error), "variance": reduce(variance),
            "logvar_floor_fraction": reduce((logvar <= -6 + 1e-6).float()),
            "logvar_ceiling_fraction": reduce((logvar >= 4 - 1e-6).float())}


def loader_for(names, entries, histories, frozen, epoch):
    dataset = T.TrainingStates(names, entries, histories, frozen.options, frozen.maximum_budget, epoch)
    return DataLoader(dataset, batch_size=8, num_workers=2, pin_memory=True,
                      generator=torch.Generator().manual_seed(SEED + epoch))


@torch.no_grad()
def estimate_constant(frozen, names, entries, histories):
    sums, count = torch.zeros(384, dtype=torch.float64, device="cuda"), 0
    for batch in loader_for(names, entries, histories, frozen, 0):
        weight = batch["weight"].cuda()
        _, mean = frozen.mean_current(batch["rgb"], batch["resolution"], weight)
        target = (batch["target"].cuda() - frozen.feature_mean) / frozen.feature_std
        normalized_weight = weight / weight.sum((1, 2, 3), keepdim=True)
        sums += (((mean - target).square() * normalized_weight).sum((0, 2, 3))).double()
        count += len(mean)
        if STOP:
            raise InterruptedError("Train-only constant fit interrupted; no fitted head emitted")
    assert count == len(names)
    return {"variance": (sums / count).float().clamp_min(1e-6).cpu(), "images": count,
            "source": "train-only frozen residual per-channel second moment; one fixed epoch, image equal",
            "mean_identity_hash": frozen.mean_identity_hash}


@torch.no_grad()
def evaluate(frozen, head, names, constant, output):
    head.eval()
    rows, image_rows = [], []
    digest = hashlib.sha256()
    keys = ("nll", "mse", "variance", "constant_nll", "logvar_floor_fraction", "logvar_ceiling_fraction")
    for name in names:
        item = torch.load(frozen.development_cache / (name + ".pt"), weights_only=True)
        target = frozen.target_for_labels(name)
        full = torch.load(frozen.target_cache / (name + ".pt"), weights_only=True)
        image_records = []
        for j in range(0, len(item["records"]), 8):
            partial = (item["features"][j:j + 8].cuda().float() - frozen.feature_mean) / frozen.feature_std
            weight = full["weight"][None].cuda().expand(len(partial), -1, -1, -1)
            resolution = item["resolution"][j:j + 8].cuda()
            mean = frozen.mean_model(partial, resolution, weight)
            digest.update(name.encode())
            digest.update(mean.cpu().contiguous().numpy().tobytes())
            logvar = head(partial, mean, resolution, weight)
            score = metrics(mean, logvar, target, weight)
            constant_logvar = constant.cuda()[None, :, None, None].log().expand_as(mean)
            score["constant_nll"] = metrics(mean, constant_logvar, target, weight)["nll"]
            assert all(torch.isfinite(value).all() for value in score.values())
            for k, record in enumerate(item["records"][j:j + 8]):
                image_records.append({"image": name, "kind": record["kind"], "budget": record["budget"],
                                      **{key: value[k].item() for key, value in score.items()}})
        image_rows.append({"image": name, **{key: float(np.mean([
            np.mean([r[key] for r in image_records if r["kind"] == kind])
            for kind in ("human", "random")])) for key in keys}})
        rows.extend(image_records)
    strata = []
    for kind in ("human", "random"):
        for budget in range(1, frozen.maximum_budget + 1):
            selected = [r for r in rows if r["kind"] == kind and r["budget"] == budget]
            if selected:
                strata.append({"kind": kind, "budget": budget, "images": len(selected),
                               **{key: float(np.mean([r[key] for r in selected])) for key in keys}})
    summary = {"images": len(names), "states": len(rows), "mean_output_sha256": digest.hexdigest(),
               "overall": {key: float(np.mean([r[key] for r in image_rows])) for key in keys},
               "aggregation": "equal available budgets within kind, equal kinds within image, equal images",
               "by_kind_budget": strata}
    P.write_json(output, {"summary": summary, "images": image_rows, "states": rows})
    return summary


def ranks(values):
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    left = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    right = np.r_[left[1:], len(values)]
    for begin, end in zip(left, right):
        result[order[begin:end]] = (begin + end - 1) / 2
    return result


def correlation(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    x, y = ranks(x), ranks(y)
    x, y = x - x.mean(), y - y.mean()
    return float((x @ y) / np.sqrt((x @ x) * (y @ y)))


def calibration_summary(states, label):
    """Candidate equal/state, budget equal/kind, kind equal/image, image equal."""
    if not states:
        return {"images": 0, "states": 0, "candidates": 0}
    cuts = np.quantile(np.concatenate([s["variance"] for s in states]), np.arange(1, 10) / 10)
    by_image = defaultdict(lambda: defaultdict(list))
    ranks_by_image = defaultdict(lambda: defaultdict(list))
    for state in states:
        pred, error = state["variance"], state["error"]
        bins = np.searchsorted(cuts, pred, side="right")
        cells = np.stack([np.bincount(bins, minlength=10),
                          np.bincount(bins, weights=pred, minlength=10),
                          np.bincount(bins, weights=error, minlength=10)], axis=1) / len(pred)
        by_image[state["image"]][state["kind"]].append(cells)
        ranks_by_image[state["image"]][state["kind"]].append(correlation(pred, error))
    names = sorted(by_image)
    values = np.asarray([np.mean([np.mean(rows, axis=0) for rows in by_image[n].values()], axis=0) for n in names])
    image_ranks = []
    for name in names:
        means = [np.mean(np.asarray(rows)[np.isfinite(rows)]) for rows in ranks_by_image[name].values()
                 if np.isfinite(rows).any()]
        image_ranks.append(float(np.mean(means)) if means else np.nan)
    image_ranks = np.asarray(image_ranks)
    average = values.mean(0)
    predicted, actual = average[:, 1].sum(), average[:, 2].sum()
    assert actual > 0
    boot = []
    indices = T.rng_for("v3_variance_calibration", label).integers(0, len(names), (2000, len(names)))
    for first in range(0, len(indices), 100):
        selected = indices[first:first + 100]
        sample = values[selected].mean(1)
        pred, err = sample[:, :, 1].sum(1), sample[:, :, 2].sum(1)
        rank_sample = image_ranks[selected]
        rank_count = np.isfinite(rank_sample).sum(1)
        rank_average = np.divide(np.nansum(rank_sample, axis=1), rank_count,
                                 out=np.full(len(selected), np.nan), where=rank_count > 0)
        boot.append(np.column_stack((pred / err, np.abs(sample[:, :, 1] - sample[:, :, 2]).sum(1) / err, rank_average)))
    boot = np.concatenate(boot)
    intervals = {}
    for i, key in enumerate(("variance_error_ratio", "normalized_decile_ece", "within_state_spearman")):
        valid = boot[:, i][np.isfinite(boot[:, i])]
        intervals[key] = np.quantile(valid, [.025, .975]).tolist() if len(valid) else None
    return {"images": len(names), "states": len(states), "candidates": sum(len(s["variance"]) for s in states),
            "mean_variance": float(predicted), "mean_actual_mse": float(actual),
            "variance_error_ratio": float(predicted / actual),
            "normalized_decile_ece": float(np.abs(average[:, 1] - average[:, 2]).sum() / actual),
            "within_state_spearman_image_macro": float(np.nanmean(image_ranks)) if np.isfinite(image_ranks).any() else None,
            "image_bootstrap95": intervals, "bootstrap_replicates": 2000,
            "decile_boundaries": cuts.tolist(),
            "deciles": [{"bin": i, "image_probability_mass": float(mass),
                         "mean_variance": float(pred / mass) if mass else None,
                         "mean_error": float(err / mass) if mass else None}
                        for i, (mass, pred, err) in enumerate(average)],
            "aggregation": "equal candidates/state, equal available budgets/kind, equal kinds/image, equal images",
            "bin_rule": "raw predicted variance quantiles fixed on development candidates; no temperature fitting"}


@torch.no_grad()
def calibrate(frozen, head, entries, run):
    states = []
    head.eval()
    with (run / "calibration_candidates.jsonl").open("w") as output:
        for index, (name, entry) in enumerate(sorted(entries.items())):
            assert entry["split"] == "development"
            obs = T.observer(entry, frozen.options)
            xy = obs.choice_grid()["xy"]
            local_weight = torch.stack([P.pool_mask(obs.local_weights(point)) for point in xy]).cuda()
            assert (local_weight.sum((1, 2, 3)) > 0).all()
            cached = torch.load(frozen.development_cache / (name + ".pt"), weights_only=True)
            target = frozen.target_for_labels(name)
            full = torch.load(frozen.target_cache / (name + ".pt"), weights_only=True)
            weight = full["weight"][None].cuda()
            for j, record in enumerate(cached["records"]):
                if record["budget"] >= frozen.maximum_budget:
                    continue
                assert record["budget"] == record["actual_budget"] == len(record["xy"])
                partial = (cached["features"][j:j + 1].cuda().float() - frozen.feature_mean) / frozen.feature_std
                resolution = cached["resolution"][j:j + 1].cuda()
                mean = frozen.mean_model(partial, resolution, weight)
                logvar = head(partial, mean, resolution, weight)
                # Channel averaging and fixed spatial averaging commute. Reduce
                # channels first to avoid an N_candidates x384x20x32 temporary.
                variance_map = logvar.exp().mean(1, keepdim=True)
                error_map = (mean - target).square().mean(1, keepdim=True)
                predicted = spatial_average(variance_map, local_weight)[:, 0].cpu().numpy().astype(np.float64)
                errors = spatial_average(error_map, local_weight)[:, 0].cpu().numpy().astype(np.float64)
                assert np.isfinite(predicted).all() and np.isfinite(errors).all() and np.all(predicted > 0)
                assert np.all(errors >= 0)
                # Eligibility is a geometry control only; retain all candidates
                # for calibration, including exact revisits/no-change actions.
                state = {"image": name, "kind": record["kind"], "budget": record["budget"],
                         "variance": predicted, "error": errors}
                states.append(state)
                for k, point in enumerate(xy):
                    output.write(json.dumps({"image": name, "kind": record["kind"], "budget": record["budget"],
                        "subject": record["subject"], "candidate_id": k, "candidate_xy": point.tolist(),
                        "variance": float(predicted[k]), "error": float(errors[k])}, allow_nan=False) + "\n")
            if (index + 1) % 25 == 0 or STOP:
                output.flush()
                print(json.dumps({"stage": "candidate_calibration", "images": index + 1}), flush=True)
            if STOP:
                raise InterruptedError("Calibration interrupted; resume terminal training checkpoint to retry calibration")
    strata = []
    for kind in ("human", "random"):
        for budget in range(1, frozen.maximum_budget):
            selected = [s for s in states if s["kind"] == kind and s["budget"] == budget]
            strata.append({"kind": kind, "budget": budget,
                           **calibration_summary(selected, f"{kind}:{budget}")})
    return {"state": "calibration_review_required", "approved": None, "calibration_pass": None,
            "overall": calibration_summary(states, "all"), "by_kind_budget": strata,
            "scope": "all371development images, all fixed-grid candidates, every available before-budget1..B-1",
            "definition": "channel-mean BEFORE variance and squared error use identical fixed soft local weights",
            "forwards": "reuse new mean partial cache encoded batch1; mean and variance heads batch1",
            "interpretation": "descriptive development calibration of selected checkpoint; no automatic acceptance"}


def main(args):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    run, mean_run = Path(args.run_dir), Path(args.mean_run).resolve()
    frozen = FrozenObserver(mean_run)
    frozen.assert_unchanged()
    assert P.digest(T.MANIFEST) == T.MANIFEST_HASH
    manifest = json.loads(T.MANIFEST.read_text())
    assert P.digest(manifest["raw_fixations"]) == T.FIXATIONS_HASH
    entries = {n: e for n, e in manifest["images"].items() if e["split"] in ("train", "development")}
    train_names = sorted(n for n, e in entries.items() if e["split"] == "train")
    dev_names = sorted(n for n, e in entries.items() if e["split"] == "development")
    assert (len(train_names), len(dev_names)) == (3343, 371)
    histories = defaultdict(list)
    for row in json.loads(Path(manifest["raw_fixations"]).read_text()):
        if row["name"] in entries:
            histories[row["name"]].append(row)
    assert frozen.target_cache == T.TARGETS
    assert frozen.cache_index["cache_id"] == frozen.config["cache_id"]
    # Verify cache content once, before fitting, including targets read directly
    # by the training Dataset rather than the guarded downstream target loader.
    for name in train_names + dev_names:
        assert P.digest(frozen.target_cache / (name + ".pt")) == frozen.cache_index["target_sha256"][name]
    for name in dev_names:
        path = frozen.development_cache / (name + ".pt")
        assert P.digest(path) == frozen.cache_index["development_sha256"][name]
        cached = torch.load(path, weights_only=True)
        assert cached["cache_id"] == frozen.config["cache_id"] and cached["image"] == name
    source_hashes = {p.name: P.digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    spec = {"mean_run": str(mean_run), "mean_checkpoint_sha256": frozen.mean_hash,
            "mean_identity_hash": frozen.mean_identity_hash,
            "frozen_mean_manifest_sha256": P.digest(mean_run / "frozen_mean.json"),
            "observer_sha256": frozen.observer_hash, "protocol_sha256": P.digest(mean_run / "protocol.json"),
            "normalization_sha256": P.digest(mean_run / "normalization.pt"), "source_sha256": source_hashes,
            "training_source_sha256": P.digest(__file__), "maximum_budget": frozen.maximum_budget,
            "common_sha256": frozen.common_hash,
            "target_cache": str(frozen.target_cache), "development_cache": str(frozen.development_cache),
            "seed": SEED, "maximum_epochs": 40, "minimum_epochs": 5, "batch_size": 8,
            "learning_rate": .0003, "weight_decay": .0001, "clip_norm": 1.,
            "early_stop_patience": 8, "relative_improvement": 1e-4,
            "lr_patience": 2, "lr_factor": .5, "lr_min": 1e-6, "variance_log_bounds": [-6, 4],
            "torch": str(torch.__version__), "cuda": torch.version.cuda, "holdout_images_used": 0}
    experiment_id = identity(spec)
    config = {**spec, "experiment_id": experiment_id, "state": "preparing", "gpu": torch.cuda.get_device_name(),
              "calibration_approved": None}
    P.write_json(run / "config.json", config)
    torch.manual_seed(SEED)
    head = VarianceHead().cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.0003, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=2,
        threshold=1e-4, threshold_mode="rel", min_lr=1e-6)
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {id(p) for p in head.parameters()}
    assert all(not p.requires_grad for p in frozen.mean_model.parameters())
    assert all(not p.requires_grad for p in frozen.encoder.parameters())
    first_epoch = first_batch = bad = 0
    best = anchor = float("inf")
    best_payload, history = None, []
    acc = {"nll_sum": 0., "images": 0, "budgets": {}}
    training_status = "training"
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        assert saved["experiment_id"] == experiment_id
        head.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        T.M.restore_rng(saved["rng"])
        first_epoch, first_batch, bad = saved["epoch"], saved["batch"], saved["bad"]
        best, anchor, best_payload = saved["best"], saved["anchor"], saved["best_snapshot"]
        history, acc, constant = saved["history"], saved["accumulator"], saved["constant"]
        training_status = saved["training_status"]
        if best_payload is not None:
            P.save(run / "best.pt", best_payload)
            P.write_json(run / "best_metrics.json", best_payload["metrics"])
    else:
        constant = estimate_constant(frozen, train_names, entries, histories)
    P.save(run / "constant_variance.pt", constant)
    initial = evaluate(frozen, head, dev_names, constant["variance"], run / "development_before_training.json")
    reference_mean_hash = initial["mean_output_sha256"]
    with (run / "curves.jsonl").open("w") as stream:
        for row in history:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    started = time.monotonic()

    def save_last(epoch, batch, status, accumulator):
        P.save(run / "last.pt", {"model": T.M.cpu_copy(head.state_dict()),
            "optimizer": T.M.cpu_copy(optimizer.state_dict()), "scheduler": scheduler.state_dict(),
            "rng": T.M.rng_state(), "epoch": epoch, "batch": batch, "best": best, "anchor": anchor,
            "bad": bad, "best_snapshot": best_payload, "history": history, "accumulator": accumulator,
            "constant": constant, "training_status": status, "experiment_id": experiment_id,
            "mean_identity_hash": frozen.mean_identity_hash, "config": dict(config)})

    config["state"] = "training"
    P.write_json(run / "config.json", config)
    if training_status == "training":
        for epoch in range(first_epoch, 40):
            names = list(np.asarray(train_names)[T.rng_for("variance_order", epoch).permutation(len(train_names))])
            loader = loader_for(names, entries, histories, frozen, epoch)
            head.train()
            if epoch != first_epoch:
                acc = {"nll_sum": 0., "images": 0, "budgets": {}}
            counts = defaultdict(int, acc["budgets"])
            for batch_index, batch in enumerate(loader):
                if epoch == first_epoch and batch_index < first_batch:
                    continue
                weight, resolution = batch["weight"].cuda(), batch["resolution"].cuda()
                partial, mean = frozen.mean_current(batch["rgb"], resolution, weight)
                target = (batch["target"].cuda() - frozen.feature_mean) / frozen.feature_std
                logvar = head(partial, mean, resolution, weight)
                scores = metrics(mean, logvar, target, weight)["nll"]
                assert torch.isfinite(scores).all()
                optimizer.zero_grad(set_to_none=True)
                scores.mean().backward()
                nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                acc["nll_sum"] += scores.detach().sum().item()
                acc["images"] += len(partial)
                for budget in batch["budget"]:
                    counts[str(budget.item())] += 1
                acc["budgets"] = dict(counts)
                if (batch_index + 1) % 100 == 0 or STOP:
                    save_last(epoch, batch_index + 1, "training", acc)
                if STOP:
                    config["state"] = "interrupted_resumable"
                    P.write_json(run / "config.json", config)
                    return 75
            assert acc["images"] == len(train_names)
            assert set(map(int, counts)) == set(range(1, frozen.maximum_budget + 1))
            summary = evaluate(frozen, head, dev_names, constant["variance"], run / f"development_epoch_{epoch + 1:03d}.json")
            assert summary["mean_output_sha256"] == reference_mean_hash, "Frozen mean outputs changed"
            frozen.assert_unchanged()
            score = summary["overall"]["nll"]
            relative = (anchor - score) / max(abs(anchor), 1e-12) if np.isfinite(anchor) else None
            if relative is None or relative >= 1e-4:
                anchor, bad = score, 0
            else:
                bad += 1
            scheduler.step(score)
            if score < best:
                best = score
                best_payload = {"model": T.M.cpu_copy(head.state_dict()), "epoch": epoch + 1,
                    "dev_nll": best, "mean_checkpoint_sha256": frozen.mean_hash,
                    "mean_identity_hash": frozen.mean_identity_hash, "normalization": frozen.normalization,
                    "metrics": summary, "config": dict(config)}
                P.save(run / "best.pt", best_payload)
                P.write_json(run / "best_metrics.json", summary)
            training_status = "early_stopped" if epoch + 1 >= 5 and bad >= 8 else "budget_exhausted" if epoch + 1 == 40 else "training"
            record = {"epoch": epoch + 1, "train_nll": acc["nll_sum"] / acc["images"], "development": summary,
                      "training_status": training_status, "lr": max(g["lr"] for g in optimizer.param_groups),
                      "bad_epochs": bad, "training_budget_counts": dict(counts)}
            history.append(record)
            with (run / "curves.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            save_last(epoch + 1, 0, training_status, {"nll_sum": 0., "images": 0, "budgets": {}})
            config.update(completed_epochs=epoch + 1, best_epoch=best_payload["epoch"], best_dev_nll=best,
                          training_status=training_status, seconds_training=time.monotonic() - started)
            P.write_json(run / "config.json", config)
            print(json.dumps({"stage": "epoch_complete", "epoch": epoch + 1, "development_nll": score,
                              "best_nll": best, "training_status": training_status}), flush=True)
            if training_status != "training":
                break
            if STOP:
                return 75
    assert best_payload is not None and training_status in ("early_stopped", "budget_exhausted")
    head.load_state_dict(best_payload["model"])
    head.eval().requires_grad_(False)
    frozen.assert_unchanged()
    config["state"] = "calibrating"
    P.write_json(run / "config.json", config)
    report = calibrate(frozen, head, {name: entries[name] for name in dev_names}, run)
    frozen.assert_unchanged()
    variance_hash = P.digest(run / "best.pt")
    report.update(mean_identity_hash=frozen.mean_identity_hash, mean_checkpoint_sha256=frozen.mean_hash,
                  variance_checkpoint_sha256=variance_hash, variance_best_epoch=best_payload["epoch"],
                  variance_best_dev_nll=best, fixed_development_best_metrics=best_payload["metrics"],
                  frozen_mean_outputs_sha256=reference_mean_hash,
                  constant_variance_sha256=P.digest(run / "constant_variance.pt"))
    P.write_json(run / "calibration_summary.json", report)
    config.update(state="calibration_review_required", training_status=training_status,
                  best_epoch=best_payload["epoch"], best_dev_nll=best, variance_checkpoint_sha256=variance_hash,
                  calibration_summary_sha256=P.digest(run / "calibration_summary.json"), calibration_approved=None)
    P.write_json(run / "config.json", config)
    binding = {"state": "calibration_review_required", "mean_run": str(mean_run), "variance_run": str(run.resolve()),
               "mean_identity_hash": frozen.mean_identity_hash, "mean_checkpoint_sha256": frozen.mean_hash,
               "variance_checkpoint_sha256": variance_hash, "maximum_budget": frozen.maximum_budget,
               "protocol_sha256": config["protocol_sha256"], "observer_sha256": frozen.observer_hash,
               "variance_source_sha256": P.digest(__file__), "common_source_sha256": source_hashes["common.py"],
               "calibration_summary_sha256": config["calibration_summary_sha256"],
               "calibration_candidates_sha256": P.digest(run / "calibration_candidates.jsonl"),
               "normalization_sha256": config["normalization_sha256"]}
    binding["observer_identity_hash"] = identity(binding)
    P.write_json(run / "observer_binding.json", binding)
    print(json.dumps({"stage": "calibration_review_required", "best_epoch": best_payload["epoch"],
                      "variance_hash": variance_hash}), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mean-run", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    try:
        status = main(args)
    except BaseException as error:
        P.write_json(Path(args.run_dir) / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    raise SystemExit(status)
