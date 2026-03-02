"""Evaluation script for comparing multiple runs."""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import wandb


def fetch_wandb_run(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """
    Fetch run data from WandB API.

    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run display name

    Returns:
        Dictionary with config, summary, and history
    """
    api = wandb.Api()
    runs = api.runs(
        f"{entity}/{project}", filters={"display_name": run_id}, order="-created_at"
    )

    if not runs:
        raise ValueError(f"No run found with display name: {run_id}")

    run = runs[0]  # Most recent run with this name

    return {
        "config": run.config,
        "summary": dict(run.summary),
        "history": run.history().to_dict("records") if hasattr(run, "history") else [],
        "name": run.name,
        "id": run.id,
        "url": run.url,
    }


def export_per_run_metrics(results_dir: Path, run_id: str, wandb_data: Dict[str, Any]):
    """
    Export metrics for a single run.

    Args:
        results_dir: Results directory
        run_id: Run identifier
        wandb_data: WandB data dictionary
    """
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Export metrics
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(wandb_data["summary"], f, indent=2)

    print(f"Exported metrics to {metrics_file}")

    # Create per-run figures if history is available
    history = wandb_data.get("history", [])
    if history:
        # Example: plot accuracy over time (if available in history)
        if any("accuracy" in h for h in history):
            fig, ax = plt.subplots(figsize=(10, 6))
            steps = [h.get("_step", i) for i, h in enumerate(history)]
            accuracies = [h.get("accuracy", 0) for h in history]
            ax.plot(steps, accuracies, marker="o")
            ax.set_xlabel("Step")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Accuracy - {run_id}")
            ax.grid(True, alpha=0.3)

            fig_file = run_dir / "accuracy_over_time.pdf"
            plt.savefig(fig_file, bbox_inches="tight")
            plt.close()
            print(f"Saved figure: {fig_file}")


def create_comparison_plots(
    results_dir: Path, run_ids: List[str], all_data: Dict[str, Dict[str, Any]]
):
    """
    Create comparison plots across runs.

    Args:
        results_dir: Results directory
        run_ids: List of run IDs
        all_data: Dictionary mapping run_id to wandb data
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Collect metrics for comparison
    metrics_to_compare = ["accuracy", "avg_tokens", "pass_b_rate", "ece", "brier_score"]

    # Create bar chart for each metric
    for metric_name in metrics_to_compare:
        values = []
        labels = []

        for run_id in run_ids:
            summary = all_data[run_id]["summary"]
            if metric_name in summary:
                values.append(summary[metric_name])
                labels.append(run_id)

        if not values:
            continue

        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        x_pos = np.arange(len(labels))

        bars = ax.bar(x_pos, values, alpha=0.7)

        # Color baseline vs proposed differently
        for i, (label, bar) in enumerate(zip(labels, bars)):
            if "comparative" in label:
                bar.set_color("gray")
            else:
                bar.set_color("steelblue")

        ax.set_xlabel("Run ID", fontsize=12)
        ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=12)
        ax.set_title(
            f"Comparison: {metric_name.replace('_', ' ').title()}", fontsize=14
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for i, (x, y) in enumerate(zip(x_pos, values)):
            ax.text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        fig_file = comparison_dir / f"comparison_{metric_name}.pdf"
        plt.savefig(fig_file, bbox_inches="tight")
        plt.close()
        print(f"Saved comparison plot: {fig_file}")

    # Create accuracy vs tokens scatter plot
    accuracies = []
    tokens = []
    labels = []

    for run_id in run_ids:
        summary = all_data[run_id]["summary"]
        if "accuracy" in summary and "avg_tokens" in summary:
            accuracies.append(summary["accuracy"])
            tokens.append(summary["avg_tokens"])
            labels.append(run_id)

    if accuracies and tokens:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Separate baseline and proposed
        for i, label in enumerate(labels):
            if "comparative" in label:
                ax.scatter(
                    tokens[i],
                    accuracies[i],
                    s=100,
                    color="red",
                    marker="s",
                    label="Baseline",
                    zorder=3,
                )
            else:
                ax.scatter(
                    tokens[i], accuracies[i], s=100, color="blue", marker="o", alpha=0.6
                )

        # Add labels to points
        for i, label in enumerate(labels):
            ax.annotate(
                label.replace("proposed-llama3-8b-gsm8k-", "").replace(
                    "comparative-1-llama3-8b-gsm8k", "baseline"
                ),
                (tokens[i], accuracies[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        ax.set_xlabel("Average Tokens per Sample", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Accuracy vs. Token Budget (Pareto Front)", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Add legend
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles[:1], legend_labels[:1])

        plt.tight_layout()
        fig_file = comparison_dir / "pareto_front_accuracy_tokens.pdf"
        plt.savefig(fig_file, bbox_inches="tight")
        plt.close()
        print(f"Saved Pareto front plot: {fig_file}")


def export_aggregated_metrics(
    results_dir: Path, run_ids: List[str], all_data: Dict[str, Dict[str, Any]]
):
    """
    Export aggregated metrics across all runs.

    Args:
        results_dir: Results directory
        run_ids: List of run IDs
        all_data: Dictionary mapping run_id to wandb data
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Collect metrics by run
    metrics_by_run = {}
    for run_id in run_ids:
        metrics_by_run[run_id] = all_data[run_id]["summary"]

    # Identify best runs
    baseline_runs = [rid for rid in run_ids if "comparative" in rid]
    proposed_runs = [rid for rid in run_ids if "proposed" in rid]

    best_baseline = None
    best_baseline_acc = -1
    if baseline_runs:
        for rid in baseline_runs:
            acc = all_data[rid]["summary"].get("accuracy", 0)
            if acc > best_baseline_acc:
                best_baseline_acc = acc
                best_baseline = rid

    best_proposed = None
    best_proposed_acc = -1
    if proposed_runs:
        for rid in proposed_runs:
            acc = all_data[rid]["summary"].get("accuracy", 0)
            if acc > best_proposed_acc:
                best_proposed_acc = acc
                best_proposed = rid

    gap = (
        best_proposed_acc - best_baseline_acc if best_proposed and best_baseline else 0
    )

    aggregated = {
        "primary_metric": "accuracy",
        "metrics_by_run": metrics_by_run,
        "best_baseline": best_baseline,
        "best_baseline_accuracy": best_baseline_acc,
        "best_proposed": best_proposed,
        "best_proposed_accuracy": best_proposed_acc,
        "gap": gap,
    }

    agg_file = comparison_dir / "aggregated_metrics.json"
    with open(agg_file, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"Exported aggregated metrics to {agg_file}")
    print(f"\nBest baseline: {best_baseline} (acc={best_baseline_acc:.3f})")
    print(f"Best proposed: {best_proposed} (acc={best_proposed_acc:.3f})")
    print(f"Gap: {gap:.3f}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate and compare multiple runs")
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Results directory"
    )
    parser.add_argument(
        "--run_ids", type=str, required=True, help="JSON list of run IDs"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default="airas", help="WandB entity"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="2026-0302-matsuzawa", help="WandB project"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    run_ids = json.loads(args.run_ids)

    print("=" * 60)
    print("Evaluation Script")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Run IDs: {run_ids}")
    print(f"WandB: {args.wandb_entity}/{args.wandb_project}")
    print("=" * 60 + "\n")

    # Fetch data for all runs
    all_data = {}
    for run_id in run_ids:
        print(f"Fetching data for run: {run_id}")
        try:
            wandb_data = fetch_wandb_run(args.wandb_entity, args.wandb_project, run_id)
            all_data[run_id] = wandb_data
            print(f"  URL: {wandb_data['url']}")
        except Exception as e:
            print(f"  ERROR: Failed to fetch run {run_id}: {e}")
            continue

    if not all_data:
        print("\nERROR: No run data fetched. Exiting.")
        return 1

    print("\n" + "=" * 60)
    print("Exporting per-run metrics and figures")
    print("=" * 60)

    for run_id, wandb_data in all_data.items():
        export_per_run_metrics(results_dir, run_id, wandb_data)

    print("\n" + "=" * 60)
    print("Creating comparison plots")
    print("=" * 60)

    create_comparison_plots(results_dir, list(all_data.keys()), all_data)

    print("\n" + "=" * 60)
    print("Exporting aggregated metrics")
    print("=" * 60)

    export_aggregated_metrics(results_dir, list(all_data.keys()), all_data)

    print("\n" + "=" * 60)
    print("Evaluation completed successfully!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
