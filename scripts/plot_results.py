"""Build public plots from aggregate records; no model or dataset access."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(root):
    data = root / "results"
    out = data / "figures"
    out.mkdir(parents=True, exist_ok=True)
    load = lambda name: json.loads((data / name).read_text())
    curves = load("mean_training.json")
    metrics = load("mean_metrics.json")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    epochs = [r["epoch"] for r in curves]
    for key, label in (("train_mse", "Training"), ("development_mse", "Development")):
        axes[0].plot(epochs, [r[key] for r in curves], label=label)
    axes[0].set(xlabel="Epoch", ylabel="Feature MSE", title="Mean predictor, r2")
    axes[0].legend(frameon=False)
    for kind, label in (("human", "Human prefixes"), ("random", "Random prefixes")):
        rows = [r for r in metrics["by_kind_budget"] if r["kind"] == kind]
        steps = [r["budget"] for r in rows]
        axes[1].plot(steps, [r["direct_mse"] for r in rows], marker=".", label=label)
        axes[2].plot(steps, [r["images"] for r in rows], marker=".", label=label)
    axes[1].set(xlabel="Observations, including initial fixation", ylabel="Feature MSE", title="Development prediction")
    axes[2].set(xlabel="Observations", ylabel="Images", title="Available development cohort")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "current_mean.png", dpi=170)
    fig.savefig(out / "current_mean.pdf")
    plt.close(fig)

    calibration = load("variance_calibration.json")["overall"]
    bins = calibration["deciles"]
    x = [r["mean_variance"] for r in bins]
    y = [r["mean_error"] for r in bins]
    fig, ax = plt.subplots(figsize=(5.2, 4))
    limits = [min(x + y) * .95, max(x + y) * 1.05]
    ax.plot(limits, limits, "--", color="gray", label="Equal predicted variance and error")
    ax.plot(x, y, "o-", color="#256a90", label="Development deciles")
    ax.set(xlabel="Mean predicted local variance", ylabel="Mean observed local squared error",
           title="Independent variance calibration, r2", xlim=limits, ylim=limits)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "current_calibration.png", dpi=170)
    fig.savefig(out / "current_calibration.pdf")
    plt.close(fig)

    historical = load("legacy_scene_prediction.json")
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in (("human", "Human"), ("U", "U"), ("localG", "Local G"),
                       ("globalG", "Global G"), ("traincenter", "Center prior"), ("uniform", "Uniform")):
        ax.plot(range(1, 9), historical["primary"][key]["mean"]["mse"], marker=".", label=label)
    ax.set(xlabel="Observations, including initial fixation", ylabel="Feature MSE",
           title="Historical hard-window observer: 364 matched images")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "legacy_scene_prediction.png", dpi=170)
    fig.savefig(out / "legacy_scene_prediction.pdf")
    plt.close(fig)
    print("Built three figures from aggregate results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    main(parser.parse_args().root)
