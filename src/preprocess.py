"""Dataset preprocessing for GSM8K math word problems."""

import re
from typing import Dict, List, Any
from datasets import load_dataset


def load_gsm8k(
    split: str = "test",
    subset: str = "main",
    num_samples: int | None = None,
    cache_dir: str = ".cache",
) -> List[Dict[str, Any]]:
    """
    Load GSM8K dataset.

    Args:
        split: Dataset split (train or test)
        subset: Dataset subset (main)
        num_samples: Number of samples to load (None for all)
        cache_dir: Cache directory for downloaded datasets

    Returns:
        List of dictionaries with 'question' and 'answer' keys
    """
    dataset = load_dataset("openai/gsm8k", subset, split=split, cache_dir=cache_dir)

    # Convert to list of dicts
    data = []
    for i, item in enumerate(dataset):
        if num_samples is not None and i >= num_samples:
            break
        data.append(
            {
                "question": item["question"],
                "answer": extract_numeric_answer(item["answer"]),
                "raw_answer": item["answer"],
            }
        )

    return data


def extract_numeric_answer(answer_text: str) -> float:
    """
    Extract numeric answer from GSM8K answer text.
    GSM8K answers are formatted as: "reasoning\n#### final_answer"

    Args:
        answer_text: Full answer text with reasoning

    Returns:
        Numeric answer as float
    """
    # GSM8K format: "reasoning\n#### final_answer"
    match = re.search(r"####\s*([-+]?[\d,]+(?:\.\d+)?)", answer_text)
    if match:
        # Remove commas and convert to float
        number_str = match.group(1).replace(",", "")
        return float(number_str)

    # Fallback: try to find any number at the end
    numbers = re.findall(r"[-+]?[\d,]+(?:\.\d+)?", answer_text)
    if numbers:
        return float(numbers[-1].replace(",", ""))

    raise ValueError(f"Could not extract numeric answer from: {answer_text}")


def extract_numeric_from_response(response: str) -> float | None:
    """
    Extract numeric answer from model response.
    Handles various formats: "42", "42.5", "$42", "42%", etc.

    Args:
        response: Model response text

    Returns:
        Numeric answer as float, or None if extraction fails
    """
    # Try to find "Draft: X" or "Final: X" format first
    for pattern in [
        r"Final:\s*([-+]?[\d,]+(?:\.\d+)?)",
        r"Draft:\s*([-+]?[\d,]+(?:\.\d+)?)",
    ]:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(",", ""))

    # Fallback: find any number in the response
    numbers = re.findall(r"[-+]?[\d,]+(?:\.\d+)?", response)
    if numbers:
        # Return the last number found
        return float(numbers[-1].replace(",", ""))

    return None


def extract_confidence(response: str) -> int:
    """
    Extract confidence level from CDP-CoT Pass A response.

    Args:
        response: Pass A response text

    Returns:
        Confidence level (0-3), or -1 if extraction fails
    """
    match = re.search(r"Confidence:\s*(\d)", response, re.IGNORECASE)
    if match:
        conf = int(match.group(1))
        if 0 <= conf <= 3:
            return conf
    return -1
