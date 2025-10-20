from __future__ import annotations

import json
import logging
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from .attention import AttentionProjections
from .calibrator import LogitCalibrator
from .config_schema import DataConfig, DecodingConfig, ExperimentConfig, MethodConfig, RunConfig
from .model_adapter import AdapterState, StepOutput, VisionLanguageAdapter

logger = logging.getLogger(__name__)


@dataclass
class GenerationSample:
    """
    Single sample to evaluate (image, prompt, reference answer, metadata).
    """

    prompt: str
    image: Optional[Any] = None
    reference: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    choices: Optional[List[str]] = None


def _temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature < 0.0:
        raise ValueError("Temperature must be non-negative.")
    if temperature == 0.0:
        # Produce a deterministic one-hot style distribution by keeping only the argmax token.
        scaled = torch.full_like(logits, float("-inf"))
        if logits.ndim == 1:
            max_idx = torch.argmax(logits)
            scaled[max_idx] = 0.0
        else:
            max_idx = torch.argmax(logits, dim=-1, keepdim=True)
            scaled.scatter_(-1, max_idx, 0.0)
        return scaled
    return logits / temperature


def _top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1].")
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    cutoff = cumulative_probs > top_p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False
    filtered_logits = logits.clone()
    filtered_logits[sorted_indices[cutoff]] = float("-inf")
    return filtered_logits


def _top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return logits
    values, _ = torch.topk(logits, top_k)
    min_values = values[..., -1, None]
    return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)


def sample_token(
    logits: torch.Tensor,
    decoding: DecodingConfig,
) -> int:
    scores = logits
    if decoding.temperature != 1.0:
        scores = _temperature_scale(scores, decoding.temperature)
    if decoding.top_k is not None:
        scores = _top_k_filtering(scores, decoding.top_k)
    if decoding.top_p < 1.0:
        scores = _top_p_filtering(scores, decoding.top_p)
    probs = F.softmax(scores, dim=-1)
    token = torch.multinomial(probs, num_samples=1).item()
    return token


def should_stop(sequence: Iterable[int], stop_words: List[str], decode_fn) -> bool:
    if not stop_words:
        return False
    decoded = decode_fn(sequence)
    return any(decoded.endswith(sw) for sw in stop_words)


def build_calibrator(config: MethodConfig) -> Optional[LogitCalibrator]:
    if not (config.use_erw or config.use_pva or config.use_ven):
        return None
    return LogitCalibrator(
        use_erw=config.use_erw,
        use_pva=config.use_pva,
        use_ven=config.use_ven,
        lambda_=config.lambda_,
        beta=config.beta,
        alpha=config.alpha,
        ven_eps=config.ven_eps,
        # Automatic scaling parameters
        auto_scale_lambda=config.auto_scale_lambda,
        target_logit_impact=config.target_logit_impact,
        max_lambda=config.max_lambda,
        min_lambda=config.min_lambda,
    )


@dataclass
class GenerationResult:
    prompt: str
    output: str
    reference: Optional[str]
    tokens: List[int]
    metadata: Dict[str, Any]


class InferencePipeline:
    def __init__(
        self,
        adapter: VisionLanguageAdapter,
        config: ExperimentConfig,
    ) -> None:
        self.adapter = adapter
        self.config = config
        self.calibrator = build_calibrator(config.method)

    def generate(self, sample: GenerationSample) -> GenerationResult:
        state = self.adapter.prepare_inputs({"prompt": sample.prompt, "image": sample.image})
        generated_tokens: List[int] = []
        metadata: Dict[str, Any] = {
            "prompt": sample.prompt,
            "method": {
                "erw": self.config.method.use_erw,
                "pva": self.config.method.use_pva,
                "ven": self.config.method.use_ven,
            "lambda": self.config.method.lambda_,
            "beta": self.config.method.beta,
            "alpha": self.config.method.alpha,
            "ven_eps": self.config.method.ven_eps,
            "dynamic_lambda": self.config.method.dynamic_lambda,
            "candidate_blending": self.config.method.candidate_blending,
            "candidate_topk": self.config.method.candidate_topk,
            "candidate_mix_cap": self.config.method.candidate_mix_cap,
            "residual_scale": self.config.method.residual_scale,
        },
    }
        if sample.metadata:
            metadata["sample"] = sample.metadata

        describe_fn = getattr(self.adapter, "describe_attention", None)
        if callable(describe_fn):
            try:
                attention_blocks = describe_fn()
                if attention_blocks:
                    metadata["attention_blocks"] = attention_blocks
            except Exception as exc:  # pragma: no cover - adapter specific
                logger.debug("Failed to fetch adapter attention description: %s", exc)

        if self.calibrator is not None:
            self.calibrator.reset()

        last_logits: Optional[torch.Tensor] = None

        for step in range(self.config.decoding.max_new_tokens):
            step_output = self.adapter.forward(state)
            logits = step_output.logits
            last_logits = logits

            if self.calibrator is not None:
                attn = step_output.attention
                logits = self.calibrator.adjust_logits(
                    logits,
                    attn,
                    token_ids=state.input_ids[0] if state.input_ids.dim() > 1 else state.input_ids,
                )

            logits = self._mask_reserved_tokens(logits)

            token = sample_token(logits, self.config.decoding)
            generated_tokens.append(token)

            if self.adapter.eos_token_id is not None and token == self.adapter.eos_token_id:
                break

            state = self.adapter.update(state, token)
            metadata.setdefault("attn_stats", []).append(
                {
                    "step": step,
                    "max_logit": float(logits.max().item()),
                    "min_logit": float(logits.min().item()),
                }
            )

            if should_stop(generated_tokens, self.config.decoding.stop_words, self.adapter.decode):
                break

        output_text = self.adapter.decode(generated_tokens)
        metadata["raw_output"] = output_text

        if sample.choices:
            enforced_output = _enforce_choices(
                output_text,
                sample.choices,
                self.adapter,
                last_logits,
            )
            metadata["enforced_output"] = enforced_output
            output_text = enforced_output

        metadata["generated_tokens"] = generated_tokens
        metadata["output_text"] = output_text

        if self.calibrator is not None:
            signal_summary = self.calibrator.get_last_signal_summary()
            if signal_summary:
                metadata["calibration_signals"] = signal_summary

                def _format_stats(stats: Optional[Dict[str, float]]) -> str:
                    if not stats:
                        return "N/A"
                    return (
                        f"mean={stats['mean']:.4f}, "
                        f"min={stats['min']:.4f}, "
                        f"max={stats['max']:.4f}"
                    )

                sample_id = None
                if sample.metadata:
                    sample_id = sample.metadata.get("dataset_index")
                prompt_preview = sample.prompt[:40].replace("\n", " ")
                logit_norm = torch.linalg.vector_norm(last_logits).item() if last_logits is not None else float("nan")
                # Get auto lambda info if available
                auto_lambda_info = ""
                if self.calibrator and self.calibrator.auto_scale_lambda:
                    auto_lambda = signal_summary.get("auto_lambda")
                    clipped = signal_summary.get("clipped")
                    dynamic_max = signal_summary.get("dynamic_max")

                    if auto_lambda:
                        clip_indicator = "🔴" if clipped and clipped['mean'] > 0.5 else "🟢"
                        max_info = f"/{dynamic_max['mean']:.1f}" if dynamic_max else ""
                        auto_lambda_info = f" | AutoLambda={auto_lambda['mean']:.2f}{max_info}{clip_indicator}"

                logger.info(
                    "[Calibrator] Sample %s | Prompt='%s...' | LogitNorm=%.4f%s | ERW[%s] PVA[%s] VEN[%s] Combined[%s]",
                    sample_id if sample_id is not None else "?",
                    prompt_preview,
                    logit_norm,
                    auto_lambda_info,
                    _format_stats(signal_summary.get("erw")),
                    _format_stats(signal_summary.get("pva")),
                    _format_stats(signal_summary.get("ven")),
                    _format_stats(signal_summary.get("combined")),
                )

        return GenerationResult(
            prompt=sample.prompt,
            output=output_text,
            reference=sample.reference,
            tokens=generated_tokens,
            metadata=metadata,
        )

    def _mask_reserved_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        reserved_ids: Optional[List[int]] = None
        if hasattr(self.adapter, "reserved_token_ids"):
            try:
                reserved_ids = self.adapter.reserved_token_ids()  # type: ignore[attr-defined]
            except Exception:
                reserved_ids = None
        if not reserved_ids:
            return logits
        masked = logits.clone()
        vocab_size = masked.shape[-1]
        for token_id in reserved_ids:
            if 0 <= token_id < vocab_size:
                masked[..., token_id] = float("-inf")
        return masked


def save_metadata(
    run_config: RunConfig,
    config: ExperimentConfig,
    results: List[GenerationResult],
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    os.makedirs(run_config.output_dir, exist_ok=True)
    payload = {
        "timestamp": time.time(),
        "run": run_config.run_name,
        "config": OmegaConf.to_container(OmegaConf.create(asdict(config)), resolve=True),
        "results": [result.metadata for result in results],
    }
    baseline_metrics = None
    if metrics is not None:
        payload["metrics"] = metrics
        baseline_metrics = _maybe_load_baseline_metrics(run_config)
        if baseline_metrics:
            deltas = {
                key: metrics.get(key, 0.0) - baseline_metrics.get(key, 0.0)
                for key in metrics.keys()
            }
            payload["delta_vs_baseline"] = deltas
    output_path = os.path.join(run_config.output_dir, run_config.metadata_filename)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def collect_samples(data_config: DataConfig) -> List[GenerationSample]:
    """
    Load evaluation samples according to the dataset configuration.
    """
    if data_config.hf_repo is not None:
        return _load_hf_dataset_samples(data_config)
    if data_config.name.lower().startswith("coco"):
        return _load_coco_samples(data_config)
    raise ValueError(
        f"No data loader configured for dataset '{data_config.name}'. "
        "Provide hf_repo or implement a custom loader."
    )


def _load_hf_dataset_samples(data_config: DataConfig) -> List[GenerationSample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets library is required for HuggingFace datasets. "
            "Install it via `pip install datasets`."
        ) from exc

    load_kwargs: Dict[str, Any] = {}
    if data_config.data_path is not None:
        cache_dir = _prepare_cache_dir(data_config.data_path)
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir

    dataset = load_dataset(
        path=data_config.hf_repo,
        name=data_config.hf_subset,
        split=data_config.split,
        **load_kwargs,
    )

    if data_config.category_filter:
        if "category" in dataset.column_names:
            dataset = dataset.filter(lambda x: x.get("category") == data_config.category_filter)
            logger.info("Filtered dataset to category '%s': %d samples", data_config.category_filter, len(dataset))
        else:
            logger.warning("category_filter specified but 'category' column not found in dataset")

    if data_config.shuffle:
        dataset = dataset.shuffle(seed=data_config.seed)
    if data_config.limit is not None:
        dataset = dataset.select(range(min(data_config.limit, len(dataset))))

    samples: List[GenerationSample] = []
    for idx, record in enumerate(dataset):
        prompt = record.get(data_config.text_column, "")
        reference = record.get(data_config.reference_column) if data_config.reference_column else None
        image = record.get(data_config.image_column)

        metadata = {
            "dataset_index": idx,
            "hf_repo": data_config.hf_repo,
            "split": data_config.split,
        }
        if data_config.hf_subset:
            metadata["hf_subset"] = data_config.hf_subset

        samples.append(
            GenerationSample(
                prompt=str(prompt),
                image=image,
                reference=str(reference) if isinstance(reference, (str, int, float)) else reference,
                metadata=metadata,
                choices=["Yes", "No"] if data_config.name.lower() == "pope" else None,
            )
        )
    return samples


def _load_coco_samples(data_config: DataConfig) -> List[GenerationSample]:
    if data_config.data_path is None:
        raise ValueError("COCO datasets require data_path.")

    split = data_config.split.lower()
    ann_filename = f"captions_{split}2017.json" if split in {"val", "train"} else f"captions_{split}.json"
    image_dir = f"{split}2017" if split in {"val", "train"} else split

    annotation_path = data_config.annotation_path or f"annotations/{ann_filename}"
    image_root = data_config.image_root or image_dir

    annotation_path = _resolve_path(data_config.data_path, annotation_path)
    image_root = _resolve_path(data_config.data_path, image_root)

    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    with open(annotation_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    image_map = {img["id"]: img for img in data.get("images", [])}
    ann_list = data.get("annotations", [])

    if data_config.shuffle:
        random.Random(data_config.seed).shuffle(ann_list)
    if data_config.limit is not None:
        ann_list = ann_list[:data_config.limit]

    samples: List[GenerationSample] = []
    for idx, ann in enumerate(ann_list):
        image_info = image_map.get(ann["image_id"])
        if not image_info:
            continue

        image_path = os.path.join(image_root, image_info["file_name"])
        prompt = ann.get(data_config.text_column, ann.get("caption", ""))
        reference = ann.get(data_config.reference_column) if data_config.reference_column else None

        samples.append(
            GenerationSample(
                prompt=str(prompt),
                image=image_path,
                reference=str(reference) if reference else None,
                metadata={
                    "dataset_index": idx,
                    "image_id": ann.get("image_id"),
                    "annotation_id": ann.get("id"),
                    "image_path": image_path,
                },
                choices=None,
            )
        )
    return samples


def _resolve_path(base: Optional[str], path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    if base is None:
        return path
    return os.path.join(base, path)


def _maybe_load_baseline_metrics(run_config: RunConfig) -> Optional[Dict[str, float]]:
    candidate_path = run_config.resume_from or os.path.join(
        os.path.dirname(run_config.output_dir), "baseline", run_config.metadata_filename
    )
    if candidate_path and os.path.exists(candidate_path):
        try:
            with open(candidate_path, "r", encoding="utf-8") as fp:
                return json.load(fp).get("metrics")
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _prepare_cache_dir(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    try:
        os.makedirs(path, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".svlm_cache_test", delete=True):
            pass
        return path
    except OSError as exc:
        logger.debug("Cache directory '%s' is not writable: %s", path, exc)
        return None


def _enforce_choices(
    raw_output: str,
    choices: List[str],
    adapter: VisionLanguageAdapter,
    logits: Optional[torch.Tensor],
) -> str:
    if not choices:
        return raw_output

    normalized_output = raw_output.strip().lower()
    for choice in choices:
        if normalized_output.startswith(choice.lower()):
            return choice

    best_choice = choices[0]
    best_logit = float("-inf")

    if logits is not None:
        for choice in choices:
            token_id = adapter.get_token_id(choice)
            if token_id is None:
                token_id = adapter.get_token_id(choice.lower())
            if token_id is None:
                continue
            if 0 <= token_id < logits.shape[0]:
                value = logits[token_id].item()
                if value > best_logit:
                    best_logit = value
                    best_choice = choice
    return best_choice
