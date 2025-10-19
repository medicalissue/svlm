from __future__ import annotations

import json
import os
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


@dataclass
class GenerationSample:
    """
    Single sample to evaluate (image, prompt, reference answer, metadata).
    """

    prompt: str
    image: Optional[Any] = None
    reference: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def _temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("Temperature must be positive.")
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
            },
        }
        if sample.metadata:
            metadata["sample"] = sample.metadata

        if self.calibrator is not None:
            self.calibrator.reset()

        for step in range(self.config.decoding.max_new_tokens):
            step_output = self.adapter.forward(state)
            logits = step_output.logits

            if self.calibrator is not None:
                attn = step_output.attention
                logits = self.calibrator.adjust_logits(logits, attn)

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
        metadata["generated_tokens"] = generated_tokens
        metadata["output_text"] = output_text

        return GenerationResult(
            prompt=sample.prompt,
            output=output_text,
            reference=sample.reference,
            tokens=generated_tokens,
            metadata=metadata,
        )


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
        "config": json.loads(OmegaConf.to_json(DictConfig({"config": asdict(config)})))["config"],
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
    Placeholder data loader. Replace with dataset-specific logic.
    """
    prompts = [
        "Describe the main objects in the image.",
        "What color is the chair?",
    ]
    samples: List[GenerationSample] = []
    max_samples = data_config.limit or len(prompts)
    for prompt in prompts[:max_samples]:
        meta = None
        if data_config.data_path is not None:
            meta = {"data_path": data_config.data_path}
        samples.append(GenerationSample(prompt=prompt, metadata=meta))
    return samples


def _maybe_load_baseline_metrics(run_config: RunConfig) -> Optional[Dict[str, float]]:
    candidate_path = None
    if run_config.resume_from is not None:
        candidate_path = run_config.resume_from
    else:
        parent_dir = os.path.dirname(run_config.output_dir)
        candidate_path = os.path.join(parent_dir, "baseline", run_config.metadata_filename)
    if candidate_path and os.path.exists(candidate_path):
        try:
            with open(candidate_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            return data.get("metrics")
        except (OSError, json.JSONDecodeError):
            return None
    return None
