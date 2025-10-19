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

DEFAULT_ADAPTER_PATH = "svlm.model_adapter.Qwen2VLAdapter"
ADAPTER_ALIASES = {
    "default": DEFAULT_ADAPTER_PATH,
    "dummy": "svlm.model_adapter.DummyAdapter",
    "baseline": "svlm.model_adapter.DummyAdapter",
    "qwen2": "svlm.model_adapter.Qwen2VLAdapter",
    "minicpm": "svlm.model_adapter.MiniCPMVAdapter",
}
MODEL_NAME_HINTS = [
    ("qwen", "svlm.model_adapter.Qwen2VLAdapter"),
    ("minicpm", "svlm.model_adapter.MiniCPMVAdapter"),
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
    adapter_cls = import_class(target_path)
    kwargs = dict(model_config.adapter_kwargs or {})
    kwargs.setdefault("model_name", model_config.name)
    if "device" not in kwargs and model_config.device is not None:
        kwargs["device"] = model_config.device
    if "torch_dtype" not in kwargs and model_config.dtype is not None:
        kwargs["torch_dtype"] = model_config.dtype
    if "revision" not in kwargs and model_config.revision is not None:
        kwargs["revision"] = model_config.revision
    if "attn_implementation" not in kwargs and model_config.attn_implementation is not None:
        kwargs["attn_implementation"] = model_config.attn_implementation
    return adapter_cls(**kwargs)


def resolve_adapter_target(model_config: ModelConfig) -> str:
    target = model_config.adapter_target

    if target and target != "auto":
        if target.lower() in ADAPTER_ALIASES:
            resolved = ADAPTER_ALIASES[target.lower()]
            log.info("Resolved adapter alias '%s' → %s", target, resolved)
            return resolved
        return target

    lowered_name = model_config.name.lower()
    for hint, path in MODEL_NAME_HINTS:
        if hint in lowered_name:
            log.info("Resolved adapter '%s' from model name hint '%s'", path, hint)
            return path

    log.info("Falling back to default adapter %s", DEFAULT_ADAPTER_PATH)
    return DEFAULT_ADAPTER_PATH


def import_class(path: str):
    module_name, class_name = path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if not module_name.startswith("src."):
            fallback = f"src.{module_name}"
            try:
                module = importlib.import_module(fallback)
                log.info("Loaded adapter via fallback path '%s'", fallback)
            except ModuleNotFoundError:
                raise exc
        else:
            raise exc
    return getattr(module, class_name)


def log_attention_structure(adapter: Any) -> None:
    describe_fn = getattr(adapter, "describe_attention", None)
    if not callable(describe_fn):
        log.debug(
            "Adapter %s does not implement describe_attention; skipping cross-attention summary.",
            adapter.__class__.__name__,
        )
        return
    try:
        blocks = describe_fn()
    except Exception as exc:  # pragma: no cover - adapter-defined behavior
        log.warning("Failed to retrieve cross-attention structure: %s", exc)
        return
    if not blocks:
        log.info("Adapter %s reported no cross-attention blocks.", adapter.__class__.__name__)
        return

    log.info("Cross-attention blocks (%d total):", len(blocks))
    for block in blocks:
        layer = block.get("layer", "?")
        block_type = block.get("type", "cross_attention")
        parts = [f"layer={layer}", f"type={block_type}"]
        if "num_heads" in block:
            parts.append(f"heads={block['num_heads']}")
        if "num_visual_tokens" in block:
            parts.append(f"visual_tokens={block['num_visual_tokens']}")
        if "hidden_size" in block:
            parts.append(f"hidden={block['hidden_size']}")
        extras = {k: v for k, v in block.items() if k not in {"layer", "type", "num_heads", "num_visual_tokens", "hidden_size"}}
        if extras:
            parts.append(f"extra={extras}")
        log.info("  - %s", ", ".join(str(p) for p in parts))


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    config = parse_config(cfg)
    log.info("Loaded configuration for run '%s'", config.run.run_name)

    # Initialize wandb if available
    wandb_run = None
    try:
        import wandb
        if wandb.run is not None:
            # Already initialized by sweep agent
            wandb_run = wandb.run
            log.info("Using existing W&B run: %s", wandb_run.name)
        else:
            # Initialize manually if not in sweep
            wandb_run = wandb.init(
                project="svlm-calibration",
                name=config.run.run_name,
                config={
                    "lambda": config.method.lambda_,
                    "beta": config.method.beta,
                    "alpha": config.method.alpha,
                    "ven_eps": config.method.ven_eps,
                    "use_erw": config.method.use_erw,
                    "use_pva": config.method.use_pva,
                    "use_ven": config.method.use_ven,
                    "temperature": config.decoding.temperature,
                    "top_p": config.decoding.top_p,
                    "data": config.data.name,
                    "model": config.model.name,
                },
                mode="online" if config.run.run_name != "baseline" else "disabled"
            )
            log.info("Initialized W&B run: %s", wandb_run.name if wandb_run else "disabled")
    except ImportError:
        log.debug("wandb not installed, skipping W&B logging")

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
    log_attention_structure(adapter)
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

        # Log metrics to wandb
        if wandb_run:
            try:
                import wandb
                # Log all numeric metrics
                wandb_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                wandb.log(wandb_metrics)
                log.info("Logged metrics to W&B: %s", wandb_metrics)
            except Exception as e:
                log.warning("Failed to log to W&B: %s", e)

    save_metadata(config.run, config, results, metrics=metrics)

    # Finish wandb run
    if wandb_run:
        try:
            import wandb
            wandb.finish()
        except:
            pass


if __name__ == "__main__":
    main()
