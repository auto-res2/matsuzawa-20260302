"""Main orchestrator for inference experiments."""


import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from src.inference import run_inference


def apply_mode_overrides(cfg: DictConfig):
    """
    Apply mode-specific overrides to configuration.

    Args:
        cfg: Configuration object
    """
    if cfg.mode == "sanity_check":
        # Reduce samples for sanity check
        if cfg.run.dataset.num_samples > 10:
            cfg.run.dataset.num_samples = 10

        # Set wandb project to sanity namespace
        if not cfg.wandb.project.endswith("-sanity"):
            cfg.wandb.project = f"{cfg.wandb.project}-sanity"

        print("Mode: sanity_check - using 10 samples and sanity namespace")

    elif cfg.mode == "main":
        # Full run with all samples
        print(f"Mode: main - using {cfg.run.dataset.num_samples} samples")

    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


def validate_sanity_check(metrics: dict):
    """
    Validate sanity check results.

    Args:
        metrics: Dictionary of metrics from inference
    """
    fail_reasons = []

    # Check minimum samples processed
    if metrics.get("total", 0) < 5:
        fail_reasons.append("insufficient_samples")

    # Check metrics are finite
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if not isinstance(value, bool) and (
                value != value or abs(value) == float("inf")
            ):
                fail_reasons.append(f"invalid_{key}")

    # Check accuracy is not 0 (unless it's genuinely all wrong)
    if metrics.get("correct", 0) == 0 and metrics.get("total", 0) > 0:
        print("WARNING: Zero correct predictions in sanity check")

    # Print validation summary
    print("\n" + "=" * 60)
    if fail_reasons:
        print(f"SANITY_VALIDATION: FAIL reason={','.join(fail_reasons)}")
    else:
        print("SANITY_VALIDATION: PASS")

    # Print summary JSON
    if "cdp_cot" in metrics:
        summary = {
            "samples": metrics.get("total", 0),
            "accuracy": metrics.get("accuracy", 0.0),
            "avg_tokens": metrics.get("avg_tokens", 0.0),
            "pass_b_rate": metrics.get("pass_b_rate", 0.0),
        }
    else:
        summary = {
            "samples": metrics.get("total", 0),
            "accuracy": metrics.get("accuracy", 0.0),
            "avg_tokens": metrics.get("avg_tokens", 0.0),
        }

    import json

    print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")
    print("=" * 60 + "\n")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for inference experiments.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 60)
    print("CDP-CoT Inference Experiment")
    print("=" * 60)
    print(f"\nRun ID: {cfg.run.run_id}")
    print(f"Method: {cfg.run.method.type}")
    print(f"Model: {cfg.run.model.name}")
    print(f"Dataset: {cfg.run.dataset.name}")
    print(f"Mode: {cfg.mode}")
    print("\n" + "=" * 60 + "\n")

    # Apply mode-specific overrides
    apply_mode_overrides(cfg)

    # Run inference
    try:
        metrics = run_inference(cfg)

        # Validate if sanity check mode
        if cfg.mode == "sanity_check":
            validate_sanity_check(metrics)

        print("\nExperiment completed successfully!")
        return 0

    except Exception as e:
        print(f"\nERROR: Experiment failed with exception: {e}")
        import traceback

        traceback.print_exc()

        if cfg.mode == "sanity_check":
            print("\n" + "=" * 60)
            print("SANITY_VALIDATION: FAIL reason=exception")
            print(f'SANITY_VALIDATION_SUMMARY: {{"error": "{str(e)}"}}')
            print("=" * 60 + "\n")

        return 1


if __name__ == "__main__":
    sys.exit(main())
