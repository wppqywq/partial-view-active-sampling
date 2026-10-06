"""Train-only budget/protocol freeze and observation previews. Run via Slurm."""
import os
if not os.environ.get("SLURM_JOB_ID"):
    raise RuntimeError("Run numerical preparation inside Slurm")

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from observer import FoveatedObserver, DEFAULT_CONFIG
from old_observer import PartialObserver as OldObserver
from rejected_observer import FoveatedObserver as RejectedObserver

MANIFEST = Path("/mnt/disk2/youyouyang/proposal2/coco_freeview/manifest.json")
MANIFEST_SHA = "7cf0f53990dad5b74657250d45197d832b5800a264c8792dc4ec3332c9541eb4"
RAW_SHA = "771167f4b5de5b9b58fb07c40d6e285d38b3b2ab9e4c6267df8a454a909da730"
OBSERVER_KWARGS = dict(DEFAULT_CONFIG)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_observer(obs, old, xy, budget):
    """Necessary interface invariants, independent of model performance."""
    assert np.array_equal(obs.target(), old.target()), "Full target preprocessing changed"
    prior = obs.observe([])
    display = prior["display_mask"]
    valid = prior["valid_mask"]
    assert valid.any()
    target = obs.target()
    for step in range(1, min(len(xy), budget) + 1):
        current = obs.observe(xy[:step])
        q = current["resolution_map"]
        assert q.dtype == np.float32 and q.shape == valid.shape
        assert np.isfinite(q).all() and (q >= 0).all() and (q <= 1).all()
        assert np.all(q[valid] >= prior["resolution_map"][valid] - 1e-7)
        assert np.isfinite(current["rgb"]).all()
        assert current["rgb"].min() >= 0 and current["rgb"].max() <= 1
        assert np.array_equal(current["rgb"][display & ~valid], target[display & ~valid])
        assert not current["rgb"][~display].any(), "Encoder padding changed"
        assert not q[~display].any()
        assert np.array_equal(current["valid_mask"], valid)
        assert np.array_equal(current["display_mask"], display)
        # No untouched source layer may contribute where sigma>=.25px.
        sigma2 = obs.blur_scale_pixels**2 * (q[valid]**-2 - 1)
        assert not current['sharp_mix_weight'][valid][sigma2 >= .25**2].any()
        prior = current
    # A repeated identical observation must change neither RGB nor resolution.
    repeated = obs.observe(np.concatenate([xy[:1], xy[:1]], axis=0))
    first = obs.observe(xy[:1])
    assert np.array_equal(repeated["rgb"], first["rgb"])
    assert np.array_equal(repeated["resolution_map"], first["resolution_map"])
    # Candidate eligibility is tied to resolution improvement, not hard masks.
    grid, _ = obs.candidates(first["resolution_map"])
    chosen = obs.observe(np.concatenate([xy[:1], grid[:1]], axis=0))
    again_grid, eligible = obs.candidates(chosen["resolution_map"])
    assert np.array_equal(again_grid, grid) and not eligible[0]
    weights = obs.local_weights(grid[0])
    assert weights.shape == valid.shape and np.isfinite(weights).all()
    assert weights.min() >= 0 and weights[valid].sum() > 0
    assert not weights[~valid].any()
    return {"steps_checked": min(len(xy), budget), "target_identical_to_old": True,
            "resolution_monotone": True, "exact_repeat_is_noop": True,
            "chosen_candidate_becomes_ineligible": True,
            "border_and_padding_preserved": True, "local_weights_valid": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert sha(MANIFEST) == MANIFEST_SHA
    manifest = json.loads(MANIFEST.read_text())
    raw = Path(manifest["raw_fixations"])
    assert sha(raw) == RAW_SHA
    train_names = {name for name, item in manifest["images"].items() if item["split"] == "train"}
    by_image = defaultdict(list)
    for row in json.loads(raw.read_text()):
        if row["name"] in train_names:
            assert len(row["X"]) == len(row["Y"]) == row["length"] > 0
            by_image[row["name"]].append(row)
    assert set(by_image) == train_names
    lengths = np.asarray([r["length"] for rows in by_image.values() for r in rows], dtype=np.int64)
    mean = float(lengths.mean())
    budget = math.ceil(mean)
    protocol = {
        "version": "continuous_foveation_v3", "job_id": os.environ["SLURM_JOB_ID"],
        "observation_revision": "r2_narrower_gaussian_scale_space",
        "revision_reason": "User rejected excessive cumulative peripheral detail in r1; no outcome-driven policy tuning",
        "maximum_budget": budget, "budget_rule": "ceil(training_trial_mean_fixation_count)",
        "budget_rationale": "Upward integer horizon at least as long as training mean; not nearest rounding.",
        "training_images": len(train_names), "training_trials": len(lengths),
        "training_fixations": int(lengths.sum()), "training_mean_fixations": mean,
        "training_median_fixations": float(np.median(lengths)),
        "nearest_integer_half_up": math.floor(mean + .5),
        "training_trials_at_least_budget": int((lengths >= budget).sum()),
        "training_length_histogram": {str(n): int((lengths == n).sum()) for n in np.unique(lengths)},
        "includes_initial_fixation": True, "human_endpoint_rule": "min(actual_trial_length,maximum_budget)",
        "human_padding": "none; retain true order, revisits, and border observations",
        "observer_kwargs": OBSERVER_KWARGS,
        "geometry": {
            "scale_basis": "full display short side, independent of image-content black borders",
            "physical_calibration": False,
            "status": "engineering scale; FreeView apparatus inheritance not directly verified",
            "freeview_verified": "Official site: same stimuli; modified Search18 procedure; no cue/response; fixed5seconds.",
            "freeview_source": "https://sites.google.com/view/cocosearch/coco-freeview",
            "search18_only_reference": "22inch,47cm,approximately54x35degrees; not transferred to FreeView as verified fact",
            "search18_source": "https://bpb-us-e1.wpmucdn.com/you.stonybrook.edu/dist/b/2177/files/2022/11/CYASHZ-2021-SR-Supplemental.pdf",
            "core_diameter_output_pixels_for_448x280": .16 * 280,
            "nominal_D50_output_pixels_for_448x280": .28 * 280,
            "D50_meaning": "engineering q equals0.5; not a measured frequency cutoff, transparency, recognition probability, or human acuity",
            "gaussian_sigma_pixels": "0.8*sqrt(q^-2-1); adjacent Gaussian layers mixed in variance",
            "scale_choice": "D50=1.75 core diameters instead of3.75; fixed single engineering revision before model outcomes",
            "untouched_source_support": "core and immediate transition where sigma<0.25px; ~3.7px beyond core with defaults",
        },
        "manifest": str(MANIFEST), "manifest_sha256": MANIFEST_SHA,
        "raw_fixations": str(raw), "raw_fixations_sha256": RAW_SHA,
        "source_sha256": {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "preview_selection": "first six pre-existing smoke_images; lowest subject ID per image; no outcome selection",
        "local_objective": "same fixed candidate-centred soft weights for localU andlocalDelta; resolution gain separate",
        "remembering": "maximum resolution across history; perfect persistent memory approximation",
        "holdout_status": "previously examined holdout is subsequent-version evaluation, not untouched confirmation",
    }
    names = [name for name in manifest["smoke_images"] if name in train_names][:6]
    assert len(names) == 6
    preview_rows = []
    with PdfPages(args.out / "observation_preview.pdf") as pdf:
        for index, name in enumerate(names):
            item = manifest["images"][name]
            assert sha(item["path"]) == item["sha256"], "Preview source image changed"
            row = min(by_image[name], key=lambda r: int(r["subject"]))
            xy = np.column_stack((row["X"], row["Y"]))
            with Image.open(item["path"]) as im:
                image = im.convert("RGB")
            obs = FoveatedObserver(image, content_box=item["display_content_box"], **OBSERVER_KWARGS)
            old = OldObserver(image, content_box=item["display_content_box"])
            rejected = RejectedObserver(image, content_box=item["display_content_box"])
            checks = check_observer(obs, old, xy, budget)
            fig, axs = plt.subplots(4, 4, figsize=(15, 10))
            steps = [1, 4, 8, budget]
            metrics = []
            for col, step in enumerate(steps):
                if len(xy) < step:
                    for ax in axs[:, col]:
                        ax.text(.5, .5, f"No observed prefix {step}\ntrial ends at {len(xy)}", ha="center", va="center")
                        ax.axis("off")
                    continue
                view, previous = obs.observe(xy[:step]), rejected.observe(xy[:step])
                instantaneous = obs.observe(xy[step-1:step])
                valid = view["valid_mask"]
                axs[0, col].imshow(previous["rgb"])
                axs[1, col].imshow(instantaneous["rgb"])
                axs[2, col].imshow(view["rgb"])
                axs[3, col].imshow(view["resolution_map"], vmin=0, vmax=1, cmap="viridis")
                points = xy[:step] * obs.scale_xy
                axs[3, col].plot(points[:, 0], points[:, 1], "w.-", lw=.7, ms=3)
                for j, (x, y) in enumerate(points):
                    axs[3, col].annotate(str(j + 1), (x, y), color="white", fontsize=6)
                axs[0, col].set_title(f"{step} observed fixation(s)")
                qmean = float(view["resolution_map"][valid].mean())
                core = float(view["core_mask"][valid].mean())
                axs[3, col].set_xlabel(f"mean q {qmean:.3f}; core {core:.1%}")
                for ax in axs[:, col]:
                    ax.set_xticks([])
                    ax.set_yticks([])
                metrics.append({"budget": step, "mean_resolution": qmean, "core_coverage": core,
                    "rejected_mean_resolution": float(previous['resolution_map'][valid].mean()),
                    "area_q_at_least_half": float((view['resolution_map'][valid] >= .5).mean()),
                    "area_any_untouched_layer_mix": float((view['sharp_mix_weight'][valid] > 0).mean()),
                    "instantaneous_mean_resolution": float(instantaneous['resolution_map'][valid].mean()),
                    "rgb_mse_to_target": float(((view['rgb'][valid]-obs.target()[valid])**2).mean()),
                    "rejected_rgb_mse_to_target": float(((previous['rgb'][valid]-obs.target()[valid])**2).mean())})
            for r, label in enumerate(("Rejected r1: cumulative", "Revised r2: CURRENT gaze only",
                                        "Revised r2: cumulative MEMORY", "Cumulative q + true history")):
                axs[r, 0].set_ylabel(label)
            fig.suptitle(f"TRAIN {name}; human subject{row['subject']}; actual length{len(xy)}\n"
                         "Same true histories. Engineering scale; not a calibrated human retina.")
            fig.tight_layout()
            pdf.savefig(fig)
            fig.savefig(args.out / f"preview_{index + 1}_{name[:-4]}.png", dpi=140)
            plt.close(fig)
            if name == '000000010083.jpg':
                view = obs.observe(xy[:8])
                current = obs.observe(xy[7:8])
                prior = rejected.observe(xy[:8])
                fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
                for ax, rgb, title in zip(axes, [obs.target(), prior['rgb'], current['rgb'], view['rgb']],
                    ['Full target (scoring only)', 'Rejected r1: cumulative 8', 'Revised r2: current gaze 8 only', 'Revised r2: memory after 8']):
                    ax.imshow(rgb); ax.set_title(title, fontsize=10); ax.axis('off')
                fig.suptitle('Same image, same actual human history. Engineering scales; no calibrated human-vision claim.', fontsize=11)
                fig.tight_layout()
                fig.savefig(args.out / 'gaze8_revision.png', dpi=150)
                plt.close(fig)
                Image.fromarray(np.rint(view['rgb']*255).astype(np.uint8)).save(args.out/'gaze8_memory.png')
                Image.fromarray(np.rint(current['rgb']*255).astype(np.uint8)).save(args.out/'gaze8_current.png')
            preview_rows.append({"image": name, "subject": row["subject"], "actual_length": len(xy),
                                 "history_xy": xy.tolist(), "checks": checks, "metrics": metrics})
    protocol["preview_records"] = preview_rows
    protocol["implementation_checks"] = "passed; source frozen; visual review still required before training"
    dump(args.out / "protocol.json", protocol)
    print(json.dumps({"status": "prepared", "maximum_budget": budget, "training_mean": mean,
                      "output": str(args.out)}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
