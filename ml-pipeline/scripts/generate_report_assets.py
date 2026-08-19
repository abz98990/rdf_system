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


MODEL_ORDER = [
    ("naive_last_day (lag_1)", "Naive\nlast day"),
    ("naive_last_week (lag_7)", "Naive\nlast week"),
    ("naive_7day_mean", "Naive\n7-day mean"),
    ("baseline_ridge", "Ridge\nbaseline"),
    ("lstm", "LSTM"),
]


def plot_fair_comparison(evaluation: dict, out_path: Path):
    """Every model on identical held-out rows, on the untouched test split --
    the comparison the report's conclusions actually rest on.
    """
    split = evaluation["test"]["models"]
    labels = [label for key, label in MODEL_ORDER if key in split]
    colors = ["#B0B0B0", "#B0B0B0", "#B0B0B0", "#4C72B0", "#DD8452"][: len(labels)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, metric in zip(axes, ["MAE", "RMSE", "MAPE"]):
        values = [split[key][metric] for key, _ in MODEL_ORDER if key in split]
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(metric + (" (%)" if metric == "MAPE" else ""))
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(
        f"Test-set performance on identical held-out rows "
        f"({evaluation['test']['n_windows']:,} windows, "
        f"{evaluation['test']['n_series']} series)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_fair_comparison_table(evaluation: dict, out_path: Path):
    lines = [
        "# Like-for-like Model Comparison",
        "",
        f"Generated: {evaluation['generated_at']}",
        "",
        evaluation["note"],
        "",
    ]
    for split_name in ["validation", "test"]:
        split = evaluation[split_name]
        lines += [
            f"## {split_name.title()} split",
            "",
            f"{split['n_windows']:,} windows across {split['n_series']} series; "
            f"mean actual sales = {split['mean_actual_sales']:.3f} units/day.",
            "",
            "| Model | MAE | RMSE | MAPE (%) |",
            "|---|---|---|---|",
        ]
        for key, label in MODEL_ORDER:
            if key in split["models"]:
                m = split["models"][key]
                clean = label.replace("\n", " ")
                lines.append(f"| {clean} | {m['MAE']:.4f} | {m['RMSE']:.4f} | {m['MAPE']:.2f} |")
        lines.append("")

    ref = evaluation.get("baseline_on_all_series_test")
    if ref:
        m = ref["metrics"]
        lines += [
            "## Reference: Ridge baseline across every series (NOT comparable)",
            "",
            f"{ref['n_rows']:,} rows across {ref['n_series']:,} series -- "
            f"MAE {m['MAE']:.4f}, RMSE {m['RMSE']:.4f}, MAPE {m['MAPE']:.2f}%.",
            "",
            ref["note"],
            "",
        ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


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

    generated = [loss_curve_path, metrics_chart_path, metrics_table_path, summary_path]

    # The like-for-like evaluation is what the report's conclusions rest on;
    # the train_pipeline numbers above score each model on its own sample.
    evaluation_path = MODELS_DIR / "final_evaluation.json"
    if evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        fair_chart_path = REPORTS_DIR / "fair_comparison.png"
        fair_table_path = REPORTS_DIR / "fair_comparison.md"
        plot_fair_comparison(evaluation, fair_chart_path)
        write_fair_comparison_table(evaluation, fair_table_path)
        generated += [fair_chart_path, fair_table_path, evaluation_path]
        print(f"Wrote {fair_chart_path}, {fair_table_path}")
    else:
        print(f"(No {evaluation_path.name} yet -- run `python cli.py evaluate` for the like-for-like comparison.)")

    FPR_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for src in generated:
        dst = FPR_ASSETS_DIR / src.name
        shutil.copy2(src, dst)
    training_log = REPORTS_DIR / "full_pipeline_log.txt"
    if training_log.exists():
        shutil.copy2(training_log, FPR_ASSETS_DIR / training_log.name)
    print(f"Copied report assets to {FPR_ASSETS_DIR}")


if __name__ == "__main__":
    run()
