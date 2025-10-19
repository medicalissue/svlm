from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Set

from .config_schema import EvaluationToggle
from .pipeline import GenerationResult


@dataclass
class MetricReport:
    metrics: Dict[str, Any]


YES_SET = {"yes", "yeah", "yep", "true", "affirmative", "correct"}
NO_SET = {"no", "nope", "nah", "false", "negative", "incorrect"}
PUNCTUATION = ".,!?;:\"'()[]{}"

# COCO 80 object categories
COCO_CATEGORIES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
}

# Common synonyms and variations for COCO objects
COCO_SYNONYMS = {
    "people": "person", "man": "person", "woman": "person", "child": "person", "boy": "person", "girl": "person",
    "bike": "bicycle", "cycle": "bicycle",
    "auto": "car", "automobile": "car", "vehicle": "car",
    "motorbike": "motorcycle",
    "plane": "airplane", "aircraft": "airplane",
    "traffic signal": "traffic light",
    "hydrant": "fire hydrant",
    "stop_sign": "stop sign",
    "meter": "parking meter",
    "seat": "bench",
    "backpack": "backpack",
    "umbrella": "umbrella",
    "bag": "handbag", "purse": "handbag",
    "necktie": "tie",
    "luggage": "suitcase",
    "skis": "skis",
    "snowboard": "snowboard",
    "ball": "sports ball",
    "kite": "kite",
    "bat": "baseball bat",
    "glove": "baseball glove",
    "skateboard": "skateboard",
    "surfboard": "surfboard",
    "racket": "tennis racket", "racquet": "tennis racket",
    "wine_glass": "wine glass", "wineglass": "wine glass",
    "cup": "cup", "mug": "cup",
    "fork": "fork",
    "knife": "knife",
    "spoon": "spoon",
    "bowl": "bowl",
    "banana": "banana",
    "apple": "apple",
    "sandwich": "sandwich",
    "orange": "orange",
    "broccoli": "broccoli",
    "carrot": "carrot",
    "hotdog": "hot dog", "hot_dog": "hot dog",
    "pizza": "pizza",
    "donut": "donut", "doughnut": "donut",
    "cake": "cake",
    "chair": "chair",
    "sofa": "couch", "couch": "couch",
    "plant": "potted plant", "potted_plant": "potted plant",
    "bed": "bed",
    "table": "dining table", "dining_table": "dining table",
    "toilet": "toilet",
    "television": "tv", "tv": "tv",
    "laptop": "laptop", "computer": "laptop",
    "mouse": "mouse",
    "remote": "remote", "remote_control": "remote",
    "keyboard": "keyboard",
    "phone": "cell phone", "cellphone": "cell phone", "cell_phone": "cell phone", "mobile": "cell phone",
    "microwave": "microwave",
    "oven": "oven",
    "toaster": "toaster",
    "sink": "sink",
    "fridge": "refrigerator", "refrigerator": "refrigerator",
    "book": "book",
    "clock": "clock",
    "vase": "vase",
    "scissors": "scissors",
    "teddy": "teddy bear", "teddy_bear": "teddy bear",
    "hairdryer": "hair drier", "hair_drier": "hair drier", "hairdrier": "hair drier",
    "toothbrush": "toothbrush"
}


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


def _extract_nouns_simple(text: str) -> Set[str]:
    """
    Simple noun extraction without spacy dependency.
    Extracts potential object words and normalizes them to COCO categories.
    """
    # Convert to lowercase and split
    text = text.lower()
    # Remove punctuation
    for p in PUNCTUATION:
        text = text.replace(p, " ")

    words = text.split()
    extracted = set()

    # Check for multi-word COCO categories first (e.g., "fire hydrant", "wine glass")
    text_lower = text.lower()
    for category in COCO_CATEGORIES:
        if " " in category and category in text_lower:
            extracted.add(category)

    # Check single words
    for word in words:
        word = word.strip()
        if not word:
            continue

        # Direct match
        if word in COCO_CATEGORIES:
            extracted.add(word)
        # Synonym match
        elif word in COCO_SYNONYMS:
            canonical = COCO_SYNONYMS[word]
            if canonical in COCO_CATEGORIES:
                extracted.add(canonical)
        # Plural forms (simple heuristic)
        elif word.endswith('s') and len(word) > 2:
            singular = word[:-1]
            if singular in COCO_CATEGORIES:
                extracted.add(singular)
            elif singular in COCO_SYNONYMS:
                canonical = COCO_SYNONYMS[singular]
                if canonical in COCO_CATEGORIES:
                    extracted.add(canonical)

    return extracted


def _extract_nouns_spacy(text: str) -> Set[str]:
    """
    Advanced noun extraction using spacy NLP.
    Falls back to simple extraction if spacy is not available.
    """
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not installed, fall back to simple extraction
            return _extract_nouns_simple(text)

        doc = nlp(text.lower())
        extracted = set()

        # Extract nouns and noun chunks
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN"):
                word = token.text.strip()
                if word in COCO_CATEGORIES:
                    extracted.add(word)
                elif word in COCO_SYNONYMS:
                    canonical = COCO_SYNONYMS[word]
                    if canonical in COCO_CATEGORIES:
                        extracted.add(canonical)

        # Also check noun chunks for multi-word objects
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            if chunk_text in COCO_CATEGORIES:
                extracted.add(chunk_text)
            elif chunk_text in COCO_SYNONYMS:
                canonical = COCO_SYNONYMS[chunk_text]
                if canonical in COCO_CATEGORIES:
                    extracted.add(canonical)

        return extracted
    except ImportError:
        # spacy not installed
        return _extract_nouns_simple(text)


def _parse_bbox(text: str) -> Optional[List[float]]:
    """
    Parse bounding box from text output.
    Supports formats:
    - [x, y, w, h]
    - <box>x1, y1, x2, y2</box>
    - x1 y1 x2 y2
    """
    # Try to find numbers in text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if len(numbers) >= 4:
        try:
            bbox = [float(n) for n in numbers[:4]]
            # Validate bbox values (should be reasonable)
            if all(0 <= b <= 1000 for b in bbox):  # Reasonable pixel range
                return bbox
        except ValueError:
            pass
    return None


def _compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute IoU (Intersection over Union) between two bounding boxes.
    Boxes can be in [x, y, w, h] or [x1, y1, x2, y2] format.
    """
    # Convert to [x1, y1, x2, y2] format if needed
    if len(box1) == 4 and len(box2) == 4:
        # Assume [x, y, w, h] if all values seem reasonable
        if box1[2] > 0 and box1[3] > 0 and box2[2] > 0 and box2[3] > 0:
            # Convert from [x, y, w, h] to [x1, y1, x2, y2]
            x1_1, y1_1, w1, h1 = box1
            x2_1, y2_1 = x1_1 + w1, y1_1 + h1

            x1_2, y1_2, w2, h2 = box2
            x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        else:
            # Already in [x1, y1, x2, y2] format
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2

        # Compute intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Compute areas
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        # Compute union
        union_area = box1_area + box2_area - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    return 0.0


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
    """
    CHAIR (Caption Hallucination Assessment with Image Relevance)

    Metrics:
    - CHAIRi: hallucinated objects / total generated objects
    - CHAIRs: captions with hallucinations / total captions

    Note: This implementation uses simple object extraction. Ground truth objects
    are expected in results[].reference as a list or comma-separated string.
    For more accurate results with COCO annotations, provide ground truth objects.
    """
    total_captions = 0
    captions_with_hallucinations = 0
    total_objects = 0
    hallucinated_objects = 0

    for result in results:
        caption = result.output
        if not caption:
            continue

        # Extract objects from generated caption
        generated_objects = _extract_nouns_simple(caption)

        if not generated_objects:
            continue

        total_captions += 1
        total_objects += len(generated_objects)

        # Get ground truth objects
        gt_objects = set()
        if result.reference:
            if isinstance(result.reference, list):
                gt_objects = {obj.lower() for obj in result.reference if obj}
            elif isinstance(result.reference, str):
                # Try to parse as comma-separated or extract objects
                gt_text = result.reference.lower()
                # If it looks like a list of objects (comma-separated)
                if ',' in gt_text:
                    gt_objects = {obj.strip() for obj in gt_text.split(',') if obj.strip()}
                else:
                    # Extract from description
                    gt_objects = _extract_nouns_simple(gt_text)

        # Normalize GT objects to COCO categories
        normalized_gt = set()
        for obj in gt_objects:
            if obj in COCO_CATEGORIES:
                normalized_gt.add(obj)
            elif obj in COCO_SYNONYMS:
                canonical = COCO_SYNONYMS[obj]
                if canonical in COCO_CATEGORIES:
                    normalized_gt.add(canonical)

        # Count hallucinations
        hallucinations_in_caption = generated_objects - normalized_gt
        if hallucinations_in_caption:
            captions_with_hallucinations += 1
            hallucinated_objects += len(hallucinations_in_caption)

    # Compute metrics
    chair_i = hallucinated_objects / total_objects if total_objects > 0 else 0.0
    chair_s = captions_with_hallucinations / total_captions if total_captions > 0 else 0.0

    return MetricReport(metrics={
        "chair_i": chair_i,
        "chair_s": chair_s,
        "total_objects": total_objects,
        "hallucinated_objects": hallucinated_objects,
        "total_captions": total_captions,
        "captions_with_hallucinations": captions_with_hallucinations
    })


def evaluate_coco(results: List[GenerationResult]) -> MetricReport:
    """
    COCO CIDEr (Consensus-based Image Description Evaluation) metric.

    Attempts to use pycocoevalcap if available.
    Install with: pip install pycocoevalcap
    """
    try:
        from pycocoevalcap.cider.cider import Cider

        gts = {}  # ground truth: {id: [ref1, ref2, ...]}
        res = {}  # results: {id: [generated]}

        for idx, result in enumerate(results):
            if result.reference is None:
                continue
            gts[idx] = [result.reference]
            res[idx] = [result.output]

        if not gts:
            return MetricReport(metrics={"coco_cider": 0.0})

        cider_scorer = Cider()
        score, scores = cider_scorer.compute_score(gts, res)

        return MetricReport(metrics={"coco_cider": float(score)})
    except ImportError:
        return MetricReport(metrics={
            "coco_cider": 0.0,
            "status": "pycocoevalcap_not_installed"
        })
    except Exception as e:
        return MetricReport(metrics={
            "coco_cider": 0.0,
            "error": str(e)
        })


def evaluate_refcoco(results: List[GenerationResult]) -> MetricReport:
    """
    RefCOCO Grounding Accuracy using IoU (Intersection over Union).

    Standard metric: Acc@0.5 (IoU > 0.5)
    Also reported: Acc@0.75, Acc@0.9

    Note: This attempts to parse bounding boxes from model output.
    Ground truth bbox is expected in results[].reference as:
    - List of 4 numbers: [x, y, w, h] or [x1, y1, x2, y2]
    - String representation of bbox
    - Dict with 'bbox' key
    """
    total = 0
    correct_at_05 = 0
    correct_at_075 = 0
    correct_at_09 = 0
    parsed_count = 0

    for result in results:
        # Parse predicted bbox from output
        pred_bbox = _parse_bbox(result.output)
        if pred_bbox is None:
            continue

        # Parse ground truth bbox
        gt_bbox = None
        if result.reference:
            if isinstance(result.reference, list) and len(result.reference) >= 4:
                try:
                    gt_bbox = [float(x) for x in result.reference[:4]]
                except (ValueError, TypeError):
                    pass
            elif isinstance(result.reference, str):
                gt_bbox = _parse_bbox(result.reference)
            elif isinstance(result.reference, dict) and 'bbox' in result.reference:
                bbox_data = result.reference['bbox']
                if isinstance(bbox_data, list) and len(bbox_data) >= 4:
                    try:
                        gt_bbox = [float(x) for x in bbox_data[:4]]
                    except (ValueError, TypeError):
                        pass

        if gt_bbox is None:
            continue

        parsed_count += 1
        total += 1

        # Compute IoU
        iou = _compute_iou(pred_bbox, gt_bbox)

        # Check accuracy at different thresholds
        if iou >= 0.5:
            correct_at_05 += 1
        if iou >= 0.75:
            correct_at_075 += 1
        if iou >= 0.9:
            correct_at_09 += 1

    # Compute accuracies
    acc_05 = correct_at_05 / total if total > 0 else 0.0
    acc_075 = correct_at_075 / total if total > 0 else 0.0
    acc_09 = correct_at_09 / total if total > 0 else 0.0

    # Mean accuracy (average of 0.5 to 0.9 in 0.05 steps)
    # For simplicity, we approximate with the three we calculated
    mean_acc = (acc_05 + acc_075 + acc_09) / 3.0

    return MetricReport(metrics={
        "refcoco_acc@0.5": acc_05,
        "refcoco_acc@0.75": acc_075,
        "refcoco_acc@0.9": acc_09,
        "refcoco_mean_acc": mean_acc,
        "total_samples": total,
        "parsed_bboxes": parsed_count
    })


def evaluate_results(toggle: EvaluationToggle, results: List[GenerationResult]) -> Dict[str, Any]:
    aggregated: Dict[str, Any] = {}
    if toggle.pope:
        aggregated.update(evaluate_pope(results).metrics)
    if toggle.coco_chair:
        aggregated.update(evaluate_chair(results).metrics)
    if toggle.cococider:
        aggregated.update(evaluate_coco(results).metrics)
    if toggle.refcoco_plus:
        aggregated.update(evaluate_refcoco(results).metrics)
    return aggregated
