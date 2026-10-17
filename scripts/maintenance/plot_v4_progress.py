"""Plot completed V4 histories without touching the live training process."""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "experiments/v4_budget_free/metrics"
OUTPUT = ROOT / "experiments/v4_budget_free/figures"
BLUE, ORANGE, GREEN, PURPLE = "#1565b8", "#d66a19", "#21864b", "#7854a6"


def main():
    sources, data = {}, {}
    for name in ("train_history", "val_history", "reference_results", "best_selection"):
        path = METRICS / (name + ".json")
        raw = path.read_bytes()
        sources[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
        data[name] = json.loads(raw)

    train_keys = (
        "epoch", "samples", "scenes", "task_loss", "weighted_comm_loss", "total_loss",
        "optional_update_ratio", "optional_probability_mean", "estimated_optional_mean_B_per_frame",
        "actual_optional_mean_B_per_frame", "soft_hard_gap_mean_B_per_frame",
        "feature_payload_BPS", "serialized_message_BPS",
    )
    val_keys = (
        "epoch", "seed", "status", "sample_count", "scene_count", "mAP", "AMOTA",
        "feature_payload_BPS", "serialized_message_BPS", "optional_update_ratio",
        "completed_utc", "checkpoint_sha256", "prediction_sha256",
    )
    train = sorted(({k: row[k] for k in train_keys} for row in data["train_history"]), key=lambda r: r["epoch"])
    val = sorted(({k: row[k] for k in val_keys} for row in data["val_history"] if row["status"] == "PASS"), key=lambda r: r["epoch"])
    refs = [row for row in data["reference_results"] if row["method"] in ("P16", "V2_MLP_B1") and row["status"] == "PASS"]
    best = max(val, key=lambda row: row["AMOTA"])
    assert train and val and all(row["sample_count"] == 3316 and row["scene_count"] == 21 for row in val)
    assert best["epoch"] == data["best_selection"]["best_epoch"]
    assert all(abs(row["task_loss"] + row["weighted_comm_loss"] - row["total_loss"]) < 1e-6 for row in train)
    assert all(abs(row["estimated_optional_mean_B_per_frame"] - row["actual_optional_mean_B_per_frame"] - row["soft_hard_gap_mean_B_per_frame"]) < 1e-6 for row in train)
    snapshot_time = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    snapshot = {
        "snapshot_time_asia_shanghai": snapshot_time,
        "source_files": sources,
        "completed_train_epochs": [row["epoch"] for row in train],
        "completed_validation_epochs": [row["epoch"] for row in val],
        "best_epoch_by_amota": best["epoch"],
        "train": train, "validation": val, "historical_references": refs,
        "interpretation": {
            "V4": "Diagnostic checkpoint trajectory at one lambda=0.05, training/inference seed=0; not a rate-distortion sweep.",
            "validation": "Official validation used for checkpoint selection; not an independent test set.",
            "historical_references": "Historical P16 and V2_MLP_B1 seed=0; unpaired; no causal superiority claim.",
            "communication": "Nominal 10Hz accounting: BPS means bytes/s, kB/s is decimal 1000 bytes/s; excludes network/protocol overhead.",
            "train_loss": "Epoch means evaluated during optimization, not frozen-checkpoint train evaluation.",
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "progress_snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
        "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.20, "grid.linewidth": 0.8,
        "figure.facecolor": "white", "axes.facecolor": "#fcfcfc", "legend.frameon": False,
        "svg.fonttype": "none",
    })
    stamp = snapshot_time[:16].replace("T", " ") + " Asia/Shanghai"
    epochs = [r["epoch"] for r in train]
    def series(rows, key, scale=1):
        return [r[key] * scale for r in rows]
    def epoch_axes(ax):
        ax.set_xlabel("Completed epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    def save(fig, name, footer, bottom=.06):
        fig.text(0.5, 0.015, footer, ha="center", va="bottom", fontsize=9, color="#4b5563")
        fig.tight_layout(rect=(0, bottom, 1, 0.93), h_pad=2.0, w_pad=2.5)
        for suffix in ("png", "svg"):
            fig.savefig(OUTPUT / (name + "." + suffix), dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(f"V4 training progress | completed epoch {epochs[-1]} | lambda = 0.05, seed = 0\nSnapshot: {stamp}", fontsize=17)
    ax = axes[0, 0]
    ax.plot(epochs, series(train, "task_loss"), "o-", color=BLUE, label="Task loss")
    ax.plot(epochs, series(train, "total_loss"), "o-", color=ORANGE, label="Total loss")
    ax.set_ylabel("Task / total loss")
    right = ax.twinx()
    right.plot(epochs, series(train, "weighted_comm_loss"), "s--", color=GREEN, label="0.05 x comm loss")
    right.set_ylabel("Weighted communication loss", color=GREEN)
    right.tick_params(axis="y", colors=GREEN)
    right.grid(False)
    lines = ax.get_lines() + right.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="upper right", fontsize=9)
    ax.set_title("Objective components (epoch means)")
    ax = axes[0, 1]
    ax.plot(epochs, series(train, "optional_update_ratio", 100), "o-", color=BLUE, label="Hard optional update rate")
    ax.plot(epochs, series(train, "optional_probability_mean", 100), "s--", color=ORANGE, label="Mean soft probability")
    ax.set_ylabel("Optional queries (%)")
    ax.set_title("Optional decisions (forced initialization excluded)")
    ax.legend()
    ax = axes[1, 0]
    ax.plot(epochs, series(train, "estimated_optional_mean_B_per_frame", .001), "o-", color=ORANGE, label="Soft expected optional cost")
    ax.plot(epochs, series(train, "actual_optional_mean_B_per_frame", .001), "o-", color=BLUE, label="Hard optional cost")
    ax.plot(epochs, series(train, "soft_hard_gap_mean_B_per_frame", .001), "s--", color=GREEN, label="Signed mean soft - hard gap")
    ax.set_ylabel("Optional latent cost (kB/frame)")
    ax.set_title("Soft surrogate and realized optional cost")
    ax.legend(fontsize=10)
    ax = axes[1, 1]
    ax.plot(epochs, series(train, "feature_payload_BPS", .001), "o-", color=BLUE, label="Feature payload")
    ax.plot(epochs, series(train, "serialized_message_BPS", .001), "o-", color=ORANGE, label="Serialized message")
    ax.set_ylabel("Communication (kB/s)")
    ax.set_title("Realized training communication at nominal 10 Hz")
    ax.legend()
    for ax in axes.flat:
        epoch_axes(ax)
    save(fig, "training_dashboard", "Training metrics are measured during optimization. Decimal kB/s = 1000 bytes/s; network overhead excluded.")

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(f"V4 validation progress | full 3316 frames / 21 scenes | best AMOTA: epoch {best['epoch']}\nCompleted validation epochs: {', '.join(str(r['epoch']) for r in val)} | Snapshot: {stamp}", fontsize=17)
    ve = [r["epoch"] for r in val]
    ax = axes[0, 0]
    ax.plot(ve, series(val, "AMOTA", 100), "o-", color=BLUE, label="AMOTA (selection metric)")
    ax.plot(ve, series(val, "mAP", 100), "s-", color=ORANGE, label="mAP")
    ax.scatter(best["epoch"], best["AMOTA"] * 100, s=230, marker="*", color=GREEN, edgecolors="white", zorder=5, label="Best AMOTA checkpoint")
    ax.set_title("Checkpoint quality (inference seed 0)")
    ax.set_ylabel("Metric (%)")
    ax.legend(fontsize=9, loc="upper left")
    ax.margins(y=.30)
    epoch_axes(ax)
    ax = axes[0, 1]
    ax.plot(ve, series(val, "feature_payload_BPS", .001), "o-", color=BLUE, label="Feature payload")
    ax.plot(ve, series(val, "serialized_message_BPS", .001), "s-", color=ORANGE, label="Serialized message")
    ax.set_title("Realized validation communication")
    ax.set_ylabel("Communication (kB/s)")
    ax.legend()
    epoch_axes(ax)
    for ax, key in zip(axes[1], ("AMOTA", "mAP")):
        ax.plot(series(val, "feature_payload_BPS", .001), series(val, key, 100), "o--", color=BLUE, label="V4 checkpoints: one lambda=0.05")
        for row in val:
            ax.annotate(f"E{row['epoch']}", (row["feature_payload_BPS"] / 1000, row[key] * 100), xytext=(7, -17 if row["epoch"] == 3 else 8), textcoords="offset points", color=BLUE, fontsize=11)
        ax.scatter(best["feature_payload_BPS"] / 1000, best[key] * 100, s=230, marker="*", color=GREEN, edgecolors="white", zorder=5, label="Best by AMOTA")
        for row, color, marker, label in zip(refs, ("#666666", PURPLE), ("^", "s"), ("P16 (historical, unpaired)", "V2 MLP-B1 (historical, unpaired)")):
            ax.scatter(row["feature_payload_BPS"] / 1000, row[key] * 100, s=110, marker=marker, color=color, edgecolors="white", zorder=4, label=label)
        ax.set_title(f"{key} vs payload: diagnostic trajectory")
        ax.set_xlabel("Feature payload (kB/s, lower is cheaper)")
        ax.set_ylabel(f"{key} (%)")
        ax.margins(x=.15, y=.35)
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .043), ncol=4, fontsize=10)
    save(fig, "validation_progress", "Official validation selects checkpoints; not independent test. Historical references are unpaired. One lambda, seed 0; no uncertainty estimate.", bottom=.10)
    print(json.dumps({
        "output": str(OUTPUT), "train_epochs": snapshot["completed_train_epochs"],
        "validation_epochs": snapshot["completed_validation_epochs"], "best_epoch": best["epoch"],
        "best_amota": best["AMOTA"], "best_map": best["mAP"],
        "best_payload_bytes_per_second": best["feature_payload_BPS"],
        "best_serialized_bytes_per_second": best["serialized_message_BPS"],
    }, indent=2))


if __name__ == "__main__":
    main()
