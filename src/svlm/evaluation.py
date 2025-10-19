from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .config_schema import EvaluationToggle
from .pipeline import GenerationResult


@dataclass
class MetricReport:
    metrics: Dict[str, float]


YES_SET = {"yes", "yeah", "yep", "true", "affirmative", "correct"}
NO_SET = {"no", "nope", "nah", "false", "negative", "incorrect"}
PUNCTUATION = ".,!?;:\"'()[]{}"


def _normalize_binary_answer(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    trimmed = text.strip().lower()
    if not trimmed:
        return None
    first_token = trimmed.split()[0].strip(PUNCTUATION)
    if first_token in YES_SET:
        return "yes"
    if first_token in NO_SET:
        return "no"
    return None


def evaluate_pope(results: List[GenerationResult]) -> MetricReport:
    total = 0
    correct = 0
    yes_count = 0
    tp = 0
    fp = 0
    fn = 0
    for result in results:
        if result.reference is None:
            continue
        pred_label = _normalize_binary_answer(result.output)
        ref_label = _normalize_binary_answer(result.reference)
        if ref_label is None:
            continue
        total += 1
        if pred_label == "yes":
            yes_count += 1
        if pred_label is None:
            if ref_label == "yes":
                fn += 1
            continue
        if pred_label == ref_label:
            correct += 1
        if pred_label == "yes" and ref_label == "yes":
            tp += 1
        elif pred_label == "yes" and ref_label == "no":
            fp += 1
        elif pred_label == "no" and ref_label == "yes":
            fn += 1
    if total == 0:
        return MetricReport(metrics={"pope_accuracy": 0.0, "pope_yes_rate": 0.0, "pope_f1": 0.0})
    accuracy = correct / total
    yes_rate = yes_count / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return MetricReport(
        metrics={
            "pope_accuracy": accuracy,
            "pope_yes_rate": yes_rate,
            "pope_f1": f1,
        }
    )


def evaluate_chair(results: List[GenerationResult]) -> MetricReport:
    # Placeholder: CHAIR requires hallucination annotations.
    return MetricReport(metrics={"chair_i": 0.0, "chair_s": 0.0})


def evaluate_coco(results: List[GenerationResult]) -> MetricReport:
    return MetricReport(metrics={"coco_cider": 0.0})


def evaluate_refcoco(results: List[GenerationResult]) -> MetricReport:
    return MetricReport(metrics={"refcoco_plus_acc": 0.0})


def evaluate_results(toggle: EvaluationToggle, results: List[GenerationResult]) -> Dict[str, float]:
    aggregated: Dict[str, float] = {}
    if toggle.pope:
        aggregated.update(evaluate_pope(results).metrics)
    if toggle.coco_chair:
        aggregated.update(evaluate_chair(results).metrics)
    if toggle.cococider:
        aggregated.update(evaluate_coco(results).metrics)
    if toggle.refcoco_plus:
        aggregated.update(evaluate_refcoco(results).metrics)
    return aggregated
