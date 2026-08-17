"""Turns models/training_summary.json (written by src.train_pipeline) into
report-ready assets for the Final Progress Report: a loss-curve plot, a
baseline-vs-LSTM metrics comparison chart, and a markdown metrics table.

Usage: python -m scripts.generate_report_assets
Outputs land in ml-pipeline/reports/ and are copied into FPR/report_assets/
so they sit next to the document that will actually cite them.
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODELS_DIR, ROOT_DIR

REPORTS_DIR = ROOT_DIR / "reports"
FPR_ASSETS_DIR = ROOT_DIR.parent / "FPR" / "report_assets"


def plot_loss_curve(history: dict, out_path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, history["train_loss"], marker="o", label="Train loss (MAE)")
    ax.plot(epochs, history["val_loss"], marker="o", label="Validation loss (MAE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE loss")
    ax.set_title("LSTM Training Progress")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metrics_comparison(baseline_metrics: dict, lstm_metrics: dict, out_path: Path):
    metric_names = ["MAE", "RMSE", "MAPE"]
    baseline_vals = [baseline_metrics[m] for m in metric_names]
    lstm_vals = [lstm_metrics[m] for m in metric_names]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, name, b_val, l_val in zip(axes, metric_names, baseline_vals, lstm_vals):
        bars = ax.bar(["Baseline\n(Ridge)", "LSTM"], [b_val, l_val], color=["#4C72B0", "#DD8452"])
        ax.set_title(name)
        ax.bar_label(bars, fmt="%.2f")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Validation Metrics: Baseline vs LSTM")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_metrics_table(summary: dict, out_path: Path):
    b = summary["baseline_val_metrics"]
    l = summary["lstm_val_metrics"]
    sizes = summary["dataset_sizes"]
    lines = [
        "# Training Metrics Summary",
        "",
        f"Generated: {summary['generated_at']}",
        f"Best model (by validation MAE): **{summary['best_model']}**",
        "",
        "## Validation Metrics",
        "",
        "| Model | MAE | RMSE | MAPE (%) |",
        "|---|---|---|---|",
        f"| Baseline (Ridge Regression) | {b['MAE']:.4f} | {b['RMSE']:.4f} | {b['MAPE']:.2f} |",
        f"| LSTM | {l['MAE']:.4f} | {l['RMSE']:.4f} | {l['MAPE']:.2f} |",
        "",
        "## Dataset Sizes",
        "",
        f"- Baseline: train={sizes['baseline_train_rows']:,} rows, "
        f"val={sizes['baseline_val_rows']:,} rows, test={sizes['baseline_test_rows']:,} rows "
        "(full filtered store)",
        f"- LSTM: {sizes['lstm_series_used']} series subsampled, "
        f"train={sizes['lstm_train_windows']:,} windows, "
        f"val={sizes['lstm_val_windows']:,} windows, "
        f"test={sizes['lstm_test_windows']:,} windows",
        "",
        "Note: the LSTM trains on a subsample of series "
        "(config.LSTM_MAX_SERIES) to keep per-epoch iteration tractable on "
        "a laptop CPU; the baseline's closed-form fit handles the full "
        "filtered store directly. See ml-pipeline/README.md.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run():
    summary_path = MODELS_DIR / "training_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} not found — run `python -m src.train_pipeline` first.")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    loss_curve_path = REPORTS_DIR / "lstm_loss_curve.png"
    metrics_chart_path = REPORTS_DIR / "metrics_comparison.png"
    metrics_table_path = REPORTS_DIR / "metrics_summary.md"

    plot_loss_curve(summary["lstm_history"], loss_curve_path)
    plot_metrics_comparison(summary["baseline_val_metrics"], summary["lstm_val_metrics"], metrics_chart_path)
    write_metrics_table(summary, metrics_table_path)
    print(f"Wrote {loss_curve_path}, {metrics_chart_path}, {metrics_table_path}")

    FPR_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for src in [loss_curve_path, metrics_chart_path, metrics_table_path, summary_path]:
        dst = FPR_ASSETS_DIR / src.name
        shutil.copy2(src, dst)
    training_log = REPORTS_DIR / "full_pipeline_log.txt"
    if training_log.exists():
        shutil.copy2(training_log, FPR_ASSETS_DIR / training_log.name)
    print(f"Copied report assets to {FPR_ASSETS_DIR}")


if __name__ == "__main__":
    run()
