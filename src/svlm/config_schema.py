from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2-VL-2B-Instruct"
    revision: Optional[str] = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    attn_implementation: Optional[str] = None
    max_sequence_length: int = 2048
    adapter_target: Optional[str] = "auto"
    adapter_kwargs: Dict[str, Any] = field(default_factory=dict)
    gpu_id: Optional[int] = None


@dataclass
class DataConfig:
    name: str = "POPE"
    split: str = "test"
    image_column: str = "image"
    text_column: str = "question"
    reference_column: Optional[str] = "answer"
    limit: Optional[int] = None
    shuffle: bool = False
    seed: int = 42
    hf_repo: Optional[str] = None
    hf_subset: Optional[str] = None
    category_filter: Optional[str] = None
    data_path: Optional[str] = None
    image_root: Optional[str] = None
    annotation_path: Optional[str] = None


@dataclass
class MethodConfig:
    use_erw: bool = True
    use_pva: bool = False
    use_ven: bool = False
    lambda_: float = 0.3
    beta: float = 0.9
    alpha: float = 0.6
    ven_eps: float = 1e-5
    dynamic_lambda: bool = False
    candidate_blending: bool = False
    candidate_topk: int = 64
    candidate_mix_cap: float = 0.95
    residual_scale: float = 0.5

    # Extreme logit manipulation modes
    use_contrastive: bool = False
    use_adversarial: bool = False
    use_adaptive_temp: bool = False

    # Contrastive sharpening parameters
    contrastive_strength: float = 10.0
    contrastive_threshold: float = 0.5

    # Adversarial perturbation parameters
    adversarial_strength: float = 0.3
    adversarial_iterations: int = 3
    adversarial_step_size: float = 0.1

    # Adaptive temperature parameters
    temp_base: float = 1.0
    temp_range: float = 5.0
    temp_visual_boost: bool = True

  # Automatic scaling parameters
    auto_scale_lambda: bool = False
    target_logit_impact: float = 0.1  # Target 10% impact on logits
    max_lambda: float = 1000.0
    min_lambda: float = 0.1


@dataclass
class DecodingConfig:
    max_new_tokens: int = 128
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: Optional[int] = None
    stop_words: List[str] = field(default_factory=list)


@dataclass
class EvaluationToggle:
    pope: bool = True
    coco_chair: bool = True
    cococider: bool = False
    refcoco_plus: bool = False


@dataclass
class RunConfig:
    run_name: str = "baseline"
    output_dir: str = "outputs"
    metadata_filename: str = "metadata.json"
    save_generations: bool = True
    resume_from: Optional[str] = None
    seed: int = 42


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    evaluation: EvaluationToggle = field(default_factory=EvaluationToggle)
    run: RunConfig = field(default_factory=RunConfig)
