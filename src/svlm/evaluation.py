from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .config_schema import EvaluationToggle
from .pipeline import GenerationResult


@dataclass
class MetricReport:
    metrics: Dict[str, float]


def evaluate_pope(results: List[GenerationResult]) -> MetricReport:
    total = 0
    correct = 0
    yes_count = 0
    for result in results:
        if result.reference is None:
            continue
        total += 1
        pred = result.output.strip().lower()
        ref = result.reference.strip().lower()
        if pred.startswith("yes"):
            yes_count += 1
        if pred.split()[:1] == ref.split()[:1]:
            correct += 1
    if total == 0:
        return MetricReport(metrics={"pope_accuracy": 0.0, "pope_yes_rate": 0.0, "pope_f1": 0.0})
    accuracy = correct / total
    yes_rate = yes_count / total
    f1 = accuracy  # Placeholder; real F1 requires detailed labels.
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
