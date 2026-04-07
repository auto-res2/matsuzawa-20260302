"""Inference script for CDP-CoT and baseline CoT experiments."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from omegaconf import DictConfig, OmegaConf
import wandb

from src.model import LLMInferenceEngine
from src.preprocess import load_gsm8k, extract_numeric_from_response, extract_confidence


def run_standard_cot(
    cfg: DictConfig, model: LLMInferenceEngine, dataset: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Run standard Chain-of-Thought inference.

    Args:
        cfg: Configuration
        model: LLM inference engine
        dataset: List of GSM8K problems

    Returns:
        Dictionary with results and metrics
    """
    results = []
    correct = 0
    total_tokens = 0

    prompt_template = cfg.run.method.prompt_template
    max_new_tokens = cfg.run.method.max_new_tokens

    for idx, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item["answer"]

        # Format prompt
        user_message = f"{question}\n\n{prompt_template}"
        prompt = model.format_chat_prompt(user_message)

        # Generate response
        output = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=cfg.run.inference.temperature,
            top_p=cfg.run.inference.top_p,
        )

        # Extract predicted answer
        predicted = extract_numeric_from_response(output["text"])
        is_correct = predicted is not None and abs(predicted - ground_truth) < 1e-6

        if is_correct:
            correct += 1

        total_tokens += output["num_tokens"]

        results.append(
            {
                "idx": idx,
                "question": question,
                "ground_truth": ground_truth,
                "response": output["text"],
                "predicted": predicted,
                "correct": is_correct,
                "num_tokens": output["num_tokens"],
            }
        )

        # Log progress
        if (idx + 1) % 10 == 0:
            print(
                f"Processed {idx + 1}/{len(dataset)} samples, "
                f"Accuracy: {correct / (idx + 1):.3f}, "
                f"Avg tokens: {total_tokens / (idx + 1):.1f}"
            )

    accuracy = correct / len(dataset)
    avg_tokens = total_tokens / len(dataset)

    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(dataset),
        "avg_tokens": avg_tokens,
        "total_tokens": total_tokens,
    }

    return {"results": results, "metrics": metrics}


def run_cdp_cot(
    cfg: DictConfig, model: LLMInferenceEngine, dataset: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Run Confidence-Calibrated Dual-Pass CoT inference.

    Args:
        cfg: Configuration
        model: LLM inference engine
        dataset: List of GSM8K problems

    Returns:
        Dictionary with results and metrics
    """
    results = []
    correct = 0
    total_tokens = 0
    pass_b_triggered = 0
    confidence_dist = {0: 0, 1: 0, 2: 0, 3: 0}

    method_cfg = cfg.run.method
    confidence_threshold = method_cfg.confidence_threshold
    max_steps = method_cfg.max_reasoning_steps

    for idx, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item["answer"]

        # Pass A: Draft answer + confidence
        pass_a_prompt = f"{question}\n\n{method_cfg.pass_a_template}"
        pass_a_formatted = model.format_chat_prompt(pass_a_prompt)

        pass_a_output = model.generate(
            pass_a_formatted,
            max_new_tokens=method_cfg.max_new_tokens_pass_a,
            temperature=cfg.run.inference.temperature,
            top_p=cfg.run.inference.top_p,
        )

        pass_a_response = pass_a_output["text"]
        confidence = extract_confidence(pass_a_response)

        # Track confidence distribution
        if confidence >= 0:
            confidence_dist[confidence] += 1

        tokens_used = pass_a_output["num_tokens"]

        # Check if Pass B is needed
        if confidence <= confidence_threshold:
            pass_b_triggered += 1

            # Pass B: Reasoning + verification
            pass_b_template = method_cfg.pass_b_template.replace(
                "{max_steps}", str(max_steps)
            )
            pass_b_prompt = (
                f"{question}\n\nYour draft was:\n{pass_a_response}\n\n{pass_b_template}"
            )
            pass_b_formatted = model.format_chat_prompt(pass_b_prompt)

            pass_b_output = model.generate(
                pass_b_formatted,
                max_new_tokens=method_cfg.max_new_tokens_pass_b,
                temperature=cfg.run.inference.temperature,
                top_p=cfg.run.inference.top_p,
            )

            final_response = pass_b_output["text"]
            tokens_used += pass_b_output["num_tokens"]
            used_pass_b = True
        else:
            final_response = pass_a_response
            used_pass_b = False

        # Extract predicted answer
        predicted = extract_numeric_from_response(final_response)
        is_correct = predicted is not None and abs(predicted - ground_truth) < 1e-6

        if is_correct:
            correct += 1

        total_tokens += tokens_used

        results.append(
            {
                "idx": idx,
                "question": question,
                "ground_truth": ground_truth,
                "pass_a_response": pass_a_response,
                "confidence": confidence,
                "used_pass_b": used_pass_b,
                "final_response": final_response,
                "predicted": predicted,
                "correct": is_correct,
                "num_tokens": tokens_used,
            }
        )

        # Log progress
        if (idx + 1) % 10 == 0:
            print(
                f"Processed {idx + 1}/{len(dataset)} samples, "
                f"Accuracy: {correct / (idx + 1):.3f}, "
                f"Avg tokens: {total_tokens / (idx + 1):.1f}, "
                f"Pass B rate: {pass_b_triggered / (idx + 1):.3f}"
            )

    accuracy = correct / len(dataset)
    avg_tokens = total_tokens / len(dataset)
    pass_b_rate = pass_b_triggered / len(dataset)

    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(dataset),
        "avg_tokens": avg_tokens,
        "total_tokens": total_tokens,
        "pass_b_triggered": pass_b_triggered,
        "pass_b_rate": pass_b_rate,
        "confidence_distribution": confidence_dist,
    }

    return {"results": results, "metrics": metrics}


def compute_calibration_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute calibration metrics (ECE and Brier score) for CDP-CoT.

    Args:
        results: List of result dictionaries with confidence and correctness

    Returns:
        Dictionary with calibration metrics
    """
    # Filter results with valid confidence scores
    valid_results = [r for r in results if "confidence" in r and r["confidence"] >= 0]

    if not valid_results:
        return {"ece": 0.0, "brier_score": 0.0}

    # Compute ECE (Expected Calibration Error)
    bins = {0: [], 1: [], 2: [], 3: []}
    for r in valid_results:
        bins[r["confidence"]].append(1.0 if r["correct"] else 0.0)

    ece = 0.0
    total_samples = len(valid_results)

    for conf_level, correct_list in bins.items():
        if correct_list:
            bin_size = len(correct_list)
            bin_accuracy = np.mean(correct_list)
            # Map confidence to probability: 0->0.25, 1->0.5, 2->0.75, 3->1.0
            bin_confidence = (conf_level + 1) / 4.0
            ece += (bin_size / total_samples) * abs(bin_accuracy - bin_confidence)

    # Compute Brier score
    brier_scores = []
    for r in valid_results:
        predicted_prob = (r["confidence"] + 1) / 4.0
        actual = 1.0 if r["correct"] else 0.0
        brier_scores.append((predicted_prob - actual) ** 2)

    brier_score = np.mean(brier_scores)

    return {"ece": ece, "brier_score": brier_score}


def run_inference(cfg: DictConfig):
    """
    Main inference function.

    Args:
        cfg: Hydra configuration
    """
    # Setup results directory
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize WandB if enabled
    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"WandB run: {wandb.run.url}")

    # Load dataset
    print(f"Loading dataset: {cfg.run.dataset.name}")
    dataset = load_gsm8k(
        split=cfg.run.dataset.split,
        subset=cfg.run.dataset.subset,
        num_samples=cfg.run.dataset.num_samples,
        cache_dir=cfg.run.inference.cache_dir,
    )
    print(f"Loaded {len(dataset)} samples")

    # Load model
    print(f"Loading model: {cfg.run.model.name}")
    model = LLMInferenceEngine(
        model_name=cfg.run.model.name,
        device=cfg.run.model.device,
        dtype=cfg.run.model.dtype,
        cache_dir=cfg.run.inference.cache_dir,
    )

    # Run inference based on method type
    method_type = cfg.run.method.type
    print(f"Running inference with method: {method_type}")

    if method_type == "standard_cot":
        output = run_standard_cot(cfg, model, dataset)
    elif method_type == "cdp_cot":
        output = run_cdp_cot(cfg, model, dataset)
        # Compute calibration metrics for CDP-CoT
        calibration = compute_calibration_metrics(output["results"])
        output["metrics"].update(calibration)
    else:
        raise ValueError(f"Unknown method type: {method_type}")

    # Save results
    results_file = results_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(output["results"], f, indent=2)

    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(output["metrics"], f, indent=2)

    print(f"\nResults saved to {results_dir}")
    print(f"Metrics: {json.dumps(output['metrics'], indent=2)}")

    # Log to WandB
    if cfg.wandb.mode != "disabled":
        wandb.log(output["metrics"])
        wandb.summary.update(output["metrics"])
        wandb.finish()

    return output["metrics"]
