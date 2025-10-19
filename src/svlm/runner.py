from __future__ import annotations

import importlib
import logging
from typing import Any, Dict

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from .config_schema import (
    DataConfig,
    DecodingConfig,
    EvaluationToggle,
    ExperimentConfig,
    MethodConfig,
    ModelConfig,
    RunConfig,
)
from .evaluation import evaluate_results
from .pipeline import InferencePipeline, GenerationResult, collect_samples, save_metadata

log = logging.getLogger(__name__)

DEFAULT_ADAPTER_PATH = "svlm.model_adapter.DummyAdapter"
ADAPTER_ALIASES = {
    "default": DEFAULT_ADAPTER_PATH,
    "dummy": DEFAULT_ADAPTER_PATH,
    "baseline": DEFAULT_ADAPTER_PATH,
}
MODEL_NAME_HINTS = [
    ("qwen", DEFAULT_ADAPTER_PATH),
    ("minicpm", DEFAULT_ADAPTER_PATH),
]


def instantiate_dataclass(target_cls, data: Dict[str, Any]):
    return target_cls(**data)


def parse_config(cfg: DictConfig) -> ExperimentConfig:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError("Hydra config must resolve to a dictionary.")
    model = instantiate_dataclass(ModelConfig, cfg_dict["model"])
    data = instantiate_dataclass(DataConfig, cfg_dict["data"])
    method = instantiate_dataclass(MethodConfig, cfg_dict["method"])
    decoding = instantiate_dataclass(DecodingConfig, cfg_dict["decoding"])
    evaluation = instantiate_dataclass(EvaluationToggle, cfg_dict.get("evaluation", {}))
    run = instantiate_dataclass(RunConfig, cfg_dict["run"])
    return ExperimentConfig(
        model=model,
        data=data,
        method=method,
        decoding=decoding,
        evaluation=evaluation,
        run=run,
    )


def load_adapter(model_config: ModelConfig):
    target_path = resolve_adapter_target(model_config)
    module_name, class_name = target_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    adapter_cls = getattr(module, class_name)
    return adapter_cls(**model_config.adapter_kwargs)


def resolve_adapter_target(model_config: ModelConfig) -> str:
    target = model_config.adapter_target
    if target is None or target == "auto":
        lowered_name = model_config.name.lower()
        for hint, path in MODEL_NAME_HINTS:
            if hint in lowered_name:
                log.info("Resolved adapter '%s' from model name hint '%s'", path, hint)
                return path
        log.info("Falling back to default adapter %s", DEFAULT_ADAPTER_PATH)
        return DEFAULT_ADAPTER_PATH

    cleaned = target.lower()
    if cleaned in ADAPTER_ALIASES:
        resolved = ADAPTER_ALIASES[cleaned]
        log.info("Resolved adapter alias '%s' → %s", target, resolved)
        return resolved

    return target


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    config = parse_config(cfg)
    log.info("Loaded configuration for run '%s'", config.run.run_name)

    if (
        config.model.gpu_id is not None
        and isinstance(config.model.device, str)
        and config.model.device.startswith("cuda")
    ):
        base_device = config.model.device.split(":", 1)[0]
        config.model.device = f"{base_device}:{config.model.gpu_id}"
        log.info("Overriding model device to %s", config.model.device)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.set_device(config.model.gpu_id)
                log.info("Set torch CUDA device to %d", config.model.gpu_id)
        except ImportError:
            log.debug("PyTorch not available; skipping torch.cuda.set_device.")
        except (AssertionError, RuntimeError) as exc:
            log.warning("Failed to set CUDA device: %s", exc)

    adapter = load_adapter(config.model)
    pipeline = InferencePipeline(adapter=adapter, config=config)

    samples = collect_samples(config.data)
    results: list[GenerationResult] = []
    for sample in samples:
        result = pipeline.generate(sample)
        results.append(result)
        log.info("Prompt: %s", result.prompt)
        log.info("Output: %s", result.output)

    config.run.output_dir = to_absolute_path(config.run.output_dir)
    metrics = evaluate_results(config.evaluation, results)
    if metrics:
        log.info("Evaluation metrics: %s", metrics)
    save_metadata(config.run, config, results, metrics=metrics)


if __name__ == "__main__":
    main()
