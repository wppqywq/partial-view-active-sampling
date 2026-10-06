"""Full early-prefix human choice: streaming conditional logit, Slurm only.

Real histories and revisits are retained. Choices are conditional on the image
content grid; border targets are counted/excluded, border history remains.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import time

if not os.environ.get("SLURM_JOB_ID"):
    raise RuntimeError("Use the supplied Slurm job")

import numpy as np
import torch
from torch.nn import functional as F
from common import FrozenObserver, P, T
from train_gain import load_gain
from signals import online_vectors, spatial_average

ROOT = Path("/mnt/disk2/youyouyang/proposal2/foveated_v3/human_choice")
REFERENCE = Path("/mnt/disk2/youyouyang/proposal2/reference_c/runs/22352/scores.csv")
REFERENCE_HASH = "cb71e6cce89f5fe0fe428b668ee0949c297e8445f057215865d52034a13b2a8f"
FEATURES = ["x", "y", "x_squared", "y_squared", "xy", "distance", "distance_squared",
            "global_resolution_increase", "fixed_local_resolution", "cell_visit_fraction", "inverse_visit_lag",
            "U", "localG", "globalG"]
MODELS = {"Controls": list(range(11)), "U": list(range(12)),
          "localG": list(range(11)) + [12], "globalG": list(range(11)) + [13]}
STOP = False


def request_stop(*_):
    global STOP
    STOP = True


signal.signal(signal.SIGUSR1, request_stop)
signal.signal(signal.SIGTERM, request_stop)


def check_stop():
    if STOP:
        raise InterruptedError("Reusable completed image/fit artifacts retained; explicit resume reruns unfinished artifact")


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def target_cell(point, grid):
    x, y = np.floor(point).astype(int)
    xe, ye = grid["x_edges"], grid["y_edges"]
    if not (xe[0] <= x < xe[-1] and ye[0] <= y < ye[-1]):
        return None
    return int((np.searchsorted(ye, y, side="right") - 1) * grid["shape"][1]
               + np.searchsorted(xe, x, side="right") - 1)


def selected_states(manifest, maximum_budget, options, run):
    """Exact reference coordinate convention is explicitly float32 canonical."""
    assert P.digest(REFERENCE) == REFERENCE_HASH
    states, counts, reference = defaultdict(list), defaultdict(int), {}
    trial_keys = set()
    for row in json.loads(Path(manifest["raw_fixations"]).read_text()):
        name, subject = row["name"], int(row["subject"])
        entry = manifest["images"][name]
        split = entry["split"]
        if split not in ("train", "development"):
            continue
        key = (name, subject)
        assert key not in trial_keys, "Repeated image-subject trial needs explicit trial identifier"
        trial_keys.add(key)
        raw = np.column_stack((row["X"], row["Y"]))
        xy = raw.astype(np.float32)
        assert np.isfinite(xy).all() and len(xy) > 0
        left, top, right, bottom = entry["display_content_box"]
        counts[split + ":trials"] += 1
        counts[split + ":unsupported_target_budgets"] += max(0, maximum_budget - len(xy))
        for t in range(1, min(len(xy), maximum_budget)):
            if not (left <= xy[t, 0] < right and top <= xy[t, 1] < bottom):
                counts[split + ":border_targets_excluded"] += 1
                continue
            record = {"image": name, "subject": subject, "target_index": t, "history_budget": t,
                      "history_xy": xy[:t].tolist(), "target_xy": xy[t].tolist(),
                      "raw_prefix_sha256": identity(raw[:t + 1].tolist())}
            states[name].append(record)
            counts[split + ":included_targets"] += 1
    wanted = {(name, r["subject"], r["target_index"]): r for name, rows in states.items()
              if manifest["images"][name]["split"] == "development" for r in rows}
    problems, grids = [], {}
    with REFERENCE.open() as stream:
        for row in csv.DictReader(stream):
            key = (row["image"], int(row["subject"]), int(row["fixation_index"]))
            if key not in wanted:
                continue
            if key in reference:
                problems.append({"key": list(key), "reason": "duplicate_reference"})
                continue
            state = wanted[key]
            if key[0] not in grids:
                grids[key[0]] = T.observer(manifest["images"][key[0]], options).choice_grid()
            expected_cell = target_cell(state["target_xy"], grids[key[0]])
            if (row["status"] != "scored" or int(row["history_length"]) != state["history_budget"]
                    or json.loads(row["history_xy"]) != state["history_xy"]
                    or [float(row["x"]), float(row["y"])] != state["target_xy"]
                    or row["target_cell_index"] == "" or int(row["target_cell_index"]) != expected_cell):
                problems.append({"key": list(key), "reason": "reference_history_target_status_mismatch"})
                continue
            reference[key] = {"DeepGaze": float(row["cell_logp"]),
                              "DG_center": float(row["centerbias_cell_logp"]),
                              "target_cell_index": int(row["target_cell_index"])}
    missing = sorted(set(wanted) - set(reference))
    P.write_json(run / "reference_alignment.json", {"reference_sha256": REFERENCE_HASH,
        "requested_development_targets": len(wanted), "matched_targets": len(reference),
        "missing": [list(key) for key in missing], "problems": problems,
        "cell_equality": "exact current content-grid equality checked for every included development target",
        "coordinate_convention": "raw JSON converted to float32, exactly as archived DeepGaze scores; raw prefix hashes retained"})
    assert not missing and not problems, "Reference targets missing/mismatched; inspect reference_alignment.json; no silent subset"
    for key, value in reference.items():
        assert np.isfinite(value["DeepGaze"]) and np.isfinite(value["DG_center"])
        wanted[key].update(value)
    protocol = {"counts": dict(counts), "maximum_budget": maximum_budget,
        "targets": "every available next fixation for actual before-budgets1..B-1; no padding",
        "aggregation": "image equal, trial equal within image, state equal within trial",
        "coordinate_convention": "canonical float32 display coordinates, matching stored reference exactly",
        "initial_fixation_scored": False, "delete_ineligible_candidates": False,
        "reference_scope": "development only; exactly all included human targets; no train reference required",
        "manifest_sha256": T.MANIFEST_HASH, "raw_fixations_sha256": T.FIXATIONS_HASH,
        "reference_sha256": REFERENCE_HASH}
    P.write_json(run / "target_protocol.json", protocol)
    return states, protocol


def state_weights(states):
    counts = defaultdict(int)
    for row in states:
        counts[int(row["subject"])] += 1
    return torch.tensor([1 / (len(counts) * counts[int(row["subject"])]) for row in states], dtype=torch.float64)


@torch.no_grad()
def image_features(name, entry, states, frozen, gain, experiment):
    obs = T.observer(entry, frozen.options)
    grid, box = obs.choice_grid(), obs.content_box
    xy = grid["xy"]
    local = torch.stack([P.pool_mask(obs.local_weights(point)) for point in xy]).cuda()
    empty = obs.observe([])
    valid = torch.from_numpy(empty["valid_mask"])[None, None].cuda()
    content = P.pool_mask(empty["valid_mask"])[None].cuda()
    # Pure geometry, computed once/image. It observes no candidate RGB/target.
    # Use the observer's exact geometric primitive: subtracting/re-adding its
    # floor would introduce rounding at the exact zero-increment boundary.
    candidate_q = torch.from_numpy(np.stack([
        obs._pad(obs._resolution_at(point * obs.scale_xy) * obs._content_mask)
        for point in xy]))[:, None].cuda()
    left, top, right, bottom = box
    normalized = 2 * (xy - [left, top]) / [right - left, bottom - top] - 1
    x, y = normalized.T
    spatial = np.column_stack((x, y, x * x, y * y, x * y))
    cell_area = np.outer(np.diff(grid["y_edges"]), np.diff(grid["x_edges"])).ravel()
    offset = np.log(cell_area / ((right - left) * (bottom - top)))
    rows, metadata = [], []
    for state_index, state in enumerate(states):
        history = state["history_xy"]
        view = obs.observe(history)
        resolution = P.pool_mask(view["resolution_map"])[None].cuda()
        partial, mean, logvar = frozen.current(P.rgb_tensor(view["rgb"])[None], resolution, content)
        current_q = torch.from_numpy(view["resolution_map"])[None, None].cuda()
        if state_index == 0:
            direct = torch.from_numpy(obs.resolution_increase(xy[0], view["resolution_map"])).cuda()
            assert torch.equal((candidate_q[0, 0] - current_q[0, 0]).clamp_min(0), direct)
        local_q = spatial_average(torch.where(content > 0, resolution / content.clamp_min(1e-12), 0.), local)[:, 0]
        values, new_mass, zero_new = [], [], []
        for start in range(0, len(xy), 16):
            sl = slice(start, start + 16)
            delta = (candidate_q[sl] - current_q).clamp_min(0)
            pooled_delta = F.avg_pool2d(delta, 14, 14)
            no_change = delta.flatten(1).amax(1) == 0
            vectors, uncertainty = online_vectors(partial, mean, logvar, content, local[sl], xy[sl], box,
                resolution, pooled_delta, len(history), frozen.maximum_budget)
            predicted_gain = gain(vectors)
            predicted_gain[no_change] = 0
            values.append(torch.column_stack((uncertainty, predicted_gain[:, 1], predicted_gain[:, 0])).cpu().numpy())
            new_mass.append((pooled_delta.sum((1, 2, 3)) / content.sum()).cpu().numpy())
            zero_new.append(no_change.cpu().numpy())
        signals, new_mass, zero_new = np.concatenate(values), np.concatenate(new_mass), np.concatenate(zero_new)
        distance = np.linalg.norm(xy - history[-1], axis=1) / min(right - left, bottom - top)
        visits, lag = np.zeros(len(xy)), np.zeros(len(xy))
        for index, point in enumerate(history):
            cell = target_cell(point, grid)
            if cell is not None:
                visits[cell] += 1
                lag[cell] = 1 / (len(history) - index)
        controls = np.column_stack((spatial, distance, distance ** 2, new_mass, local_q.cpu().numpy(),
                                    visits / len(history), lag))
        value = np.column_stack((controls, signals)).astype(np.float32)
        assert value.shape == (len(xy), len(FEATURES)) and np.isfinite(value).all()
        rows.append(value)
        cell = target_cell(state["target_xy"], grid)
        assert cell is not None
        if "target_cell_index" in state:
            assert cell == state["target_cell_index"], "Reference cell differs from exact current content grid"
        metadata.append({**state, "target_cell_index": cell, "candidate_count": len(xy),
                         "target_previously_visited_cell": bool(visits[cell] > 0),
                         "target_zero_resolution_increase": bool(zero_new[cell]),
                         "zero_change_candidate_count": int(zero_new.sum())})
        check_stop()
    return {"image": name, "experiment": experiment, "features": torch.from_numpy(np.stack(rows)),
            "offset": torch.from_numpy(offset), "states": metadata, "state_weights": state_weights(metadata)}


def feature_normalization(names, cache):
    total, square = torch.zeros(len(FEATURES), dtype=torch.float64), torch.zeros(len(FEATURES), dtype=torch.float64)
    for name in names:
        item = torch.load(cache / (name + ".pt"), weights_only=True)
        values, weight = item["features"].double(), item["state_weights"]
        assert torch.isclose(weight.sum(), weight.new_tensor(1.))
        total += (values.mean(1) * weight[:, None]).sum(0)
        square += (values.square().mean(1) * weight[:, None]).sum(0)
        check_stop()
    mean = total / len(names)
    std = (square / len(names) - mean.square()).clamp_min(0).sqrt()
    constant = std < 1e-8
    std[constant] = 1
    return {"mean": mean, "std": std, "constant_features": [FEATURES[i] for i in torch.where(constant)[0].tolist()],
            "features": FEATURES, "source": "train image->trial->state->candidate equal-weight moments", "images": len(names)}


def fit(names, cache, norm, model_name, artifact, experiment):
    if artifact.exists():
        result = json.loads(artifact.read_text())
        assert result["experiment"] == experiment and result["model"] == model_name
        return result
    columns = MODELS[model_name]
    beta = torch.zeros(len(columns), dtype=torch.float64, device="cuda", requires_grad=True)
    optimizer = torch.optim.LBFGS([beta], lr=1., max_iter=60, max_eval=80,
        tolerance_grad=1e-6, tolerance_change=1e-12, line_search_fn="strong_wolfe")
    calls = 0
    def closure():
        nonlocal calls
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for name in names:
            check_stop()
            item = torch.load(cache / (name + ".pt"), weights_only=True)
            # One image is resident at a time; candidates/states never globally packed.
            x = ((item["features"].double() - norm["mean"]) / norm["std"])[..., columns].cuda()
            y = torch.tensor([r["target_cell_index"] for r in item["states"]], device="cuda")
            logits = x @ beta + item["offset"].cuda()[None]
            loss = ((torch.logsumexp(logits, 1) - logits.gather(1, y[:, None])[:, 0])
                    * item["state_weights"].cuda()).sum() / len(names)
            loss.backward()
            total += loss.item()
        penalty = .0005 * beta.square().sum()
        penalty.backward()
        total += penalty.item()
        calls += 1
        assert np.isfinite(total) and torch.isfinite(beta.grad).all()
        print(json.dumps({"stage": "fit", "model": model_name, "closure": calls, "objective": total}), flush=True)
        return beta.new_tensor(total)
    optimizer.step(closure)
    objective = closure().item()
    gradient = beta.grad.abs().max().item()
    coefficients = beta.detach().cpu().tolist()
    result = {"experiment": experiment, "model": model_name, "columns": columns,
              "features": [FEATURES[i] for i in columns], "coefficients": coefficients,
              "objective": objective, "gradient_infinity_norm": gradient,
              "iterations": int(optimizer.state[beta]["n_iter"]), "closure_calls": calls,
              "status": "optimized" if gradient <= 1e-6 else "optimization_incomplete",
              "signal_coefficient": coefficients[-1] if model_name != "Controls" else None,
              "signal_interpretation": "positive conditional coefficient prefers higher signal; negative prefers lower",
              "training_aggregation": "image->trial->state macro; L2=.001 on train-standardized coefficients"}
    P.write_json(artifact, result)
    return result


def macro_summary(rows, label, models):
    """Bootstrap images, retaining trial/state averaging inside each image."""
    if not rows:
        return {"images": 0, "states": 0}
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["image"]][row["subject"]].append(row)
    names = sorted(grouped)
    matrix = np.array([[np.mean([np.mean([r[model] for r in trial]) for trial in grouped[name].values()])
                        for model in models] for name in names])
    samples = T.rng_for("v3_human_choice", label).integers(0, len(names), (2000, len(names)))
    boot = np.concatenate([matrix[samples[i:i + 100]].mean(1) for i in range(0, len(samples), 100)])
    result = {"images": len(names), "trials": sum(len(v) for v in grouped.values()), "states": len(rows),
              "aggregation": "image->trial->state macro", "bootstrap_replicates": 2000, "models": {}, "pairs": []}
    for j, model in enumerate(models):
        result["models"][model] = {"macro_logp": float(matrix[:, j].mean()),
            "image_bootstrap95": np.quantile(boot[:, j], [.025, .975]).tolist()}
    for first, second in (("localG", "U"), ("globalG", "U"), ("U", "Controls"),
                          ("localG", "Controls"), ("globalG", "Controls"), ("DeepGaze", "localG")):
        if first in models and second in models:
            a, b = models.index(first), models.index(second)
            result["pairs"].append({"first": first, "second": second,
                "macro_logp_difference": float((matrix[:, a] - matrix[:, b]).mean()),
                "image_bootstrap95": np.quantile(boot[:, a] - boot[:, b], [.025, .975]).tolist(),
                "direction": "positive favors first; descriptive intervals, not multiplicity corrected"})
    return result


@torch.no_grad()
def evaluate(names, cache, norm, fits, split, run, maximum_budget):
    rows = []
    with (run / f"{split}_predictions.jsonl").open("w") as output:
        for name in names:
            item = torch.load(cache / (name + ".pt"), weights_only=True)
            x = ((item["features"].double() - norm["mean"]) / norm["std"]).cuda()
            y = torch.tensor([r["target_cell_index"] for r in item["states"]], device="cuda")
            values = {}
            for model, fit_result in fits.items():
                beta = torch.tensor(fit_result["coefficients"], device="cuda", dtype=torch.float64)
                logits = x[..., fit_result["columns"]] @ beta + item["offset"].cuda()[None]
                logp = logits.gather(1, y[:, None])[:, 0] - torch.logsumexp(logits, 1)
                assert torch.isfinite(logp).all()
                values[model] = logp.cpu().tolist()
            for index, meta in enumerate(item["states"]):
                row = {key: meta[key] for key in ("image", "subject", "target_index", "history_budget", "target_cell_index",
                                                "target_previously_visited_cell", "target_zero_resolution_increase")}
                row.update({model: values[model][index] for model in fits})
                if split == "development":
                    row.update({model: meta[model] for model in ("DeepGaze", "DG_center")})
                rows.append(row)
                output.write(json.dumps(row, allow_nan=False) + "\n")
            check_stop()
    models = list(fits) + (["DeepGaze", "DG_center"] if split == "development" else [])
    summary = {"split": split, "overall": macro_summary(rows, split, models),
        "by_budget": [{"history_budget": budget,
            **macro_summary([r for r in rows if r["history_budget"] == budget], f"{split}:{budget}", models)}
            for budget in range(1, maximum_budget)],
        "reference_targets": "same complete target subset, image/subject/index/cell/history/coordinate equality verified" if split == "development" else "not evaluated",
        "conditional_support": "fixed content grid; border target observations excluded; all content candidates retained"}
    P.write_json(run / f"{split}_summary.json", summary)
    return summary


def main(args):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    run = Path(args.run_dir)
    frozen = FrozenObserver(args.mean_run, args.variance_run, require_calibration=True)
    gain = load_gain(args.gain_run, frozen)
    states, target_protocol = selected_states(frozen.manifest, frozen.maximum_budget, frozen.options, run)
    entries = frozen.manifest["images"]
    train_names = sorted(n for n in states if entries[n]["split"] == "train")
    dev_names = sorted(n for n in states if entries[n]["split"] == "development")
    assert (len(train_names), len(dev_names)) == (3343, 371)
    spec = {"observer_identity": frozen.identity, "observer_identity_hash": frozen.identity_hash,
            "gain_checkpoint_sha256": gain.best_hash, "gain_normalization_sha256": gain.norm_hash,
            "sources": {p.name: P.digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            "target_protocol": target_protocol, "features": FEATURES, "models": MODELS,
            "l2": .001, "lbfgs_max_iterations": 60, "lbfgs_max_evaluations": 80,
            "streaming": "one image per GPU batch and disk artifact; completed images/models reused exactly",
            "no_change_boundary": "G=0 iff candidate adds exactly zero resolution; no candidate deletion",
            "training": "full3343train only; allactual beforebudgets1..B-1; no synthetic histories",
            "holdout_images_used": 0, "reference_sha256": REFERENCE_HASH,
            "limitations": ["descriptive development evaluation; upstream checkpoints selected on development",
                "gain train features are in-sample to upstream observer/G",
                "conditional content-grid likelihood excludes border targets",
                "freeview humans saw normal full images; offline model uses simulated cumulative foveation",
                "controls statistically adjust spatial/history/q features; not causal matching",
                "canonical float32 coordinates preserve archived DeepGaze reference convention",
                "no model-recovery sweep; coefficients do not establish unique psychological mechanisms"]}
    experiment = identity(spec)
    cache = ROOT / "features" / experiment
    cache.mkdir(parents=True, exist_ok=True)
    config = {**spec, "experiment": experiment, "cache": str(cache), "state": "extracting_features"}
    P.write_json(run / "config.json", config)
    started = time.monotonic()
    hashes = {}
    for index, name in enumerate(train_names + dev_names):
        artifact = cache / (name + ".pt")
        if artifact.exists():
            item = torch.load(artifact, weights_only=True)
            assert item["experiment"] == experiment and item["image"] == name
            assert len(item["states"]) == len(states[name])
            for stored, expected in zip(item["states"], states[name]):
                assert all(stored[key] == value for key, value in expected.items())
        else:
            assert P.digest(entries[name]["path"]) == entries[name]["sha256"]
            P.save(artifact, image_features(name, entries[name], states[name], frozen, gain, experiment))
        hashes[name] = P.digest(artifact)
        if (index + 1) % 10 == 0 or index + 1 == len(states):
            progress = {"stage": "features", "images": index + 1, "total": len(states),
                        "seconds": time.monotonic() - started}
            P.write_json(run / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        check_stop()
    frozen.assert_unchanged()
    gain.assert_unchanged()
    P.write_json(run / "feature_index.json", {"experiment": experiment, "images": hashes})
    norm = feature_normalization(train_names, cache)
    norm["experiment"] = experiment
    P.save(run / "normalization.pt", norm)
    config.update(state="fitting", normalization_sha256=P.digest(run / "normalization.pt"))
    P.write_json(run / "config.json", config)
    fit_dir = cache / "fits"
    fit_dir.mkdir(exist_ok=True)
    fits = {model: fit(train_names, cache, norm, model, fit_dir / (model + ".json"), experiment) for model in MODELS}
    P.write_json(run / "fits.json", fits)
    config["state"] = "evaluating"
    P.write_json(run / "config.json", config)
    train = evaluate(train_names, cache, norm, fits, "train", run, frozen.maximum_budget)
    dev = evaluate(dev_names, cache, norm, fits, "development", run, frozen.maximum_budget)
    frozen.assert_unchanged()
    gain.assert_unchanged()
    config.update(state="completed", optimization_complete=all(r["status"] == "optimized" for r in fits.values()),
                  seconds=time.monotonic() - started, development_summary_sha256=P.digest(run / "development_summary.json"),
                  train_summary_sha256=P.digest(run / "train_summary.json"),
                  fits_sha256=P.digest(run / "fits.json"), feature_index_sha256=P.digest(run / "feature_index.json"))
    P.write_json(run / "config.json", config)
    print(json.dumps({"stage": "completed", "optimization_complete": config["optimization_complete"],
                      "development": dev["overall"]}), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mean-run", required=True)
    parser.add_argument("--variance-run", required=True)
    parser.add_argument("--gain-run", required=True)
    args = parser.parse_args()
    try:
        status = main(args)
    except BaseException as error:
        P.write_json(Path(args.run_dir) / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    raise SystemExit(status)
