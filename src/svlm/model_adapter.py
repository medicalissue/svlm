from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol

import torch

from .attention import AttentionProjections

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None


@dataclass
class StepOutput:
    logits: torch.Tensor
    attention: AttentionProjections
    past_key_values: Any


@dataclass
class AdapterState:
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    pixel_values: Optional[torch.Tensor] = None
    image_sizes: Optional[torch.Tensor] = None
    image_grid_thw: Optional[Any] = None
    image_token_index: Optional[Any] = None
    step: int = 0
    past_key_values: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class VisionLanguageAdapter(Protocol):
    """
    Abstraction over a VLM model capable of stepwise generation.
    """

    def prepare_inputs(self, sample: Dict[str, Any]) -> AdapterState:
        ...

    def forward(
        self,
        state: AdapterState,
    ) -> StepOutput:
        ...

    def update(
        self,
        state: AdapterState,
        token_id: int,
    ) -> AdapterState:
        ...

    def decode(self, token_ids: Iterable[int]) -> str:
        ...

    @property
    def eos_token_id(self) -> Optional[int]:
        ...

    def describe_attention(self) -> Optional[List[Dict[str, Any]]]:
        ...

    def get_token_id(self, text: str) -> Optional[int]:
        ...


class DummyAdapter(VisionLanguageAdapter):
    """
    Adapter that generates random logits/attentions for debugging.
    """

    def __init__(self, vocab_size: int = 32000, num_heads: int = 4, num_visual_tokens: int = 32):
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_visual_tokens = num_visual_tokens
        self._device = torch.device("cpu")

    def prepare_inputs(self, sample: Dict[str, Any]) -> AdapterState:
        prompt = sample.get("prompt", "")
        tokens = torch.randint(0, self.vocab_size, (len(prompt.split()) or 1,), device=self._device)
        return AdapterState(input_ids=tokens.unsqueeze(0))

    def forward(self, state: AdapterState) -> StepOutput:
        logits = torch.randn(self.vocab_size, device=self._device)
        text_length = state.input_ids.shape[1] + state.step
        attention = AttentionProjections(
            cross_modal=torch.rand(self.num_heads, self.num_visual_tokens, text_length + 1, device=self._device),
            self_attn=torch.rand(self.num_heads, text_length + 1, text_length + 1, device=self._device),
        )
        return StepOutput(logits=logits, attention=attention, past_key_values=None)

    def update(self, state: AdapterState, token_id: int) -> AdapterState:
        new_token = torch.tensor([[token_id]], device=self._device)
        new_input = torch.cat([state.input_ids, new_token], dim=1)
        return AdapterState(
            input_ids=new_input,
            attention_mask=None,
            pixel_values=None,
            image_sizes=None,
            step=state.step + 1,
            past_key_values=None,
        )

    def decode(self, token_ids: Iterable[int]) -> str:
        return " ".join(str(tid) for tid in token_ids)

    @property
    def eos_token_id(self) -> Optional[int]:
        return None

    def describe_attention(self) -> Optional[List[Dict[str, Any]]]:
        return [
            {
                "layer": idx,
                "type": "cross_attention",
                "num_heads": self.num_heads,
                "num_visual_tokens": self.num_visual_tokens,
                "hidden_size": None,
                "note": "Dummy adapter does not reflect a real model.",
            }
            for idx in range(2)
        ]

    def get_token_id(self, text: str) -> Optional[int]:
        return None


def _ensure_pil_image(image: Any) -> Optional["Image.Image"]:
    if image is None:
        return None
    if Image is None:
        raise ImportError("Pillow is required for image handling. Install it via `pip install Pillow`.")
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    if isinstance(image, bytes):
        from io import BytesIO

        return Image.open(BytesIO(image)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def _as_torch_dtype(dtype: Optional[str]) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        normalized = dtype.lower()
        if normalized in {"float16", "fp16"}:
            return torch.float16
        if normalized in {"float32", "fp32"}:
            return torch.float32
        if normalized in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if normalized in {"float64", "fp64"}:
            return torch.float64
        raise ValueError(f"Unsupported dtype string '{dtype}'.")
    raise TypeError(f"Unsupported dtype type {type(dtype)}.")


class HuggingFaceVLAdapter(VisionLanguageAdapter):
    """
    Generic HuggingFace VLM adapter that relies on AutoProcessor + AutoModelForCausalLM.

    Subclasses customize tokenizer decode logic or cross-attention extraction as needed.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: Optional[str] = None,
        revision: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = True,
        max_new_tokens: int = 128,
        model_loader: Optional[Callable[..., Any]] = None,
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "transformers is required for HuggingFace adapters. "
                "Install it via `pip install transformers accelerate`."
            ) from exc

        self.device = torch.device(device)
        self.torch_dtype = _as_torch_dtype(torch_dtype)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        processor_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "revision": revision,
            "torch_dtype": self.torch_dtype,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:  # pragma: no cover
                raise ImportError("AutoTokenizer unavailable; install transformers.") from exc
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                revision=revision,
            )

        model_kwargs["output_attentions"] = True
        model_kwargs["use_cache"] = False  # we recompute full pass for attention matrices

        if model_loader is None:
            model_loader = AutoModelForCausalLM.from_pretrained

        self.model = model_loader(
            model_name,
            **model_kwargs,
        )
        self.model.to(self.device)
        self.model.eval()

        if hasattr(self.model.config, "output_attentions"):
            self.model.config.output_attentions = True
        if hasattr(self.model.config, "output_cross_attentions"):
            self.model.config.output_cross_attentions = True
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        self._vision_config = getattr(self.model.config, "vision_config", None)
        self._text_config = getattr(self.model.config, "text_config", self.model.config)

    def prepare_inputs(self, sample: Dict[str, Any]) -> AdapterState:
        prompt = sample.get("prompt", "")
        image = _ensure_pil_image(sample.get("image"))

        processor_inputs = self.processor(
            text=[prompt],
            images=[image] if image is not None else None,
            return_tensors="pt",
        )

        processor_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in processor_inputs.items()}
        input_ids = processor_inputs.get("input_ids")
        attention_mask = processor_inputs.get("attention_mask")
        pixel_values = processor_inputs.get("pixel_values")
        image_sizes = processor_inputs.get("image_sizes")
        image_grid_thw = processor_inputs.get("image_grid_thw")
        image_token_index = processor_inputs.get("image_token_index")

        return AdapterState(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            image_grid_thw=image_grid_thw,
            image_token_index=image_token_index,
            step=0,
            past_key_values=None,
            metadata={"prompt": prompt},
        )

    def forward(self, state: AdapterState) -> StepOutput:
        with torch.no_grad():
            outputs = self.model(
                input_ids=state.input_ids,
                attention_mask=state.attention_mask,
                pixel_values=state.pixel_values,
                image_sizes=state.image_sizes,
                image_grid_thw=state.image_grid_thw,
                image_token_index=state.image_token_index,
                output_attentions=True,
                output_cross_attentions=True,
                use_cache=False,
            )

        logits = outputs.logits[0, -1, :]
        attention = self._build_attention(outputs)
        return StepOutput(
            logits=logits,
            attention=attention,
            past_key_values=None,
        )

    def update(self, state: AdapterState, token_id: int) -> AdapterState:
        token_tensor = torch.tensor([[token_id]], device=state.input_ids.device, dtype=state.input_ids.dtype)
        new_input_ids = torch.cat([state.input_ids, token_tensor], dim=1)

        if state.attention_mask is not None:
            mask_token = torch.ones((1, 1), device=state.attention_mask.device, dtype=state.attention_mask.dtype)
            new_attention = torch.cat([state.attention_mask, mask_token], dim=1)
        else:
            new_attention = None

        return AdapterState(
            input_ids=new_input_ids,
            attention_mask=new_attention,
            pixel_values=state.pixel_values,
            image_sizes=state.image_sizes,
            image_grid_thw=state.image_grid_thw,
            image_token_index=state.image_token_index,
            step=state.step + 1,
            past_key_values=None,
            metadata=state.metadata,
        )

    def decode(self, token_ids: Iterable[int]) -> str:
        if self.tokenizer is None:
            return " ".join(str(tid) for tid in token_ids)
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)

    @property
    def eos_token_id(self) -> Optional[int]:
        if self.tokenizer is not None and hasattr(self.tokenizer, "eos_token_id"):
            return self.tokenizer.eos_token_id
        if hasattr(self.model.config, "eos_token_id"):
            return self.model.config.eos_token_id
        return None

    def describe_attention(self) -> Optional[List[Dict[str, Any]]]:
        blocks: List[Dict[str, Any]] = []
        num_layers = getattr(self._text_config, "num_hidden_layers", None)
        num_heads = getattr(self._text_config, "num_attention_heads", None)
        if num_layers is not None:
            for idx in range(num_layers):
                blocks.append(
                    {
                        "layer": idx,
                        "type": "self_attention",
                        "num_heads": num_heads,
                        "hidden_size": getattr(self._text_config, "hidden_size", None),
                        "note": "Transformer text block",
                    }
                )
        if self._vision_config is not None:
            blocks.append(
                {
                    "layer": "vision",
                    "type": "vision_encoder",
                    "num_heads": getattr(self._vision_config, "num_attention_heads", None),
                    "hidden_size": getattr(self._vision_config, "hidden_size", None),
                    "image_size": getattr(self._vision_config, "image_size", None),
                    "note": "Vision encoder configuration",
                }
            )
        cross_info = getattr(self.model.config, "cross_attention_layers", None)
        if cross_info:
            for item in cross_info:
                if isinstance(item, dict):
                    blocks.append({"type": "cross_attention", **item})
        if not blocks:
            return None
        return blocks

    def get_token_id(self, text: str) -> Optional[int]:
        if self.tokenizer is None:
            return None
        candidate_strings = [
            text,
            text.lower(),
            text.upper(),
            text.capitalize(),
            f" {text}",
            f" {text.lower()}",
            f" {text.upper()}",
            f" {text.capitalize()}",
        ]
        seen = set()
        for candidate in candidate_strings:
            if candidate in seen:
                continue
            seen.add(candidate)
            tokens = self.tokenizer.encode(candidate, add_special_tokens=False)
            if tokens:
                return tokens[0]
        return None

    def _build_attention(self, outputs) -> AttentionProjections:
        cross_attn = getattr(outputs, "cross_attentions", None)
        self_attn = getattr(outputs, "attentions", None)

        cross_tensor = None
        if cross_attn:
            # take last layer cross attention for batch index 0
            last_cross = cross_attn[-1][0]  # (num_heads, query_len, kv_len)
            cross_tensor = last_cross.permute(0, 2, 1).contiguous()

        self_tensor = None
        if self_attn:
            last_self = self_attn[-1][0]  # (num_heads, query_len, seq_len)
            self_tensor = last_self

        if cross_tensor is None:
            cross_tensor = torch.zeros(
                (self_tensor.shape[0], 1, self_tensor.shape[-1]) if self_tensor is not None else (1, 1, 1),
                device=outputs.logits.device,
            )
        if self_tensor is None:
            self_tensor = torch.zeros(
                (cross_tensor.shape[0], cross_tensor.shape[-1], cross_tensor.shape[-1]),
                device=outputs.logits.device,
            )

        return AttentionProjections(
            cross_modal=cross_tensor,
            self_attn=self_tensor,
        )


class Qwen2VLAdapter(HuggingFaceVLAdapter):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "cuda",
        torch_dtype: Optional[str] = "bfloat16",
        revision: Optional[str] = None,
        attn_implementation: Optional[str] = None,
    ) -> None:
        try:
            from transformers import Qwen2VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Qwen2VLForConditionalGeneration not found. Ensure transformers>=4.41 and trust_remote_code enabled."
            ) from exc
        super().__init__(
            model_name=model_name,
            device=device,
            torch_dtype=torch_dtype,
            revision=revision,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
            model_loader=Qwen2VLForConditionalGeneration.from_pretrained,
        )

    def describe_attention(self) -> Optional[List[Dict[str, Any]]]:
        base = super().describe_attention() or []
        base.append(
            {
                "layer": "multimodal",
                "type": "cross_attention",
                "num_heads": getattr(self.model.config, "multi_modal_hidden_size", None),
                "note": "Qwen2-VL multimodal fusion layers (approximate)",
            }
        )
        return base

    def prepare_inputs(self, sample: Dict[str, Any]) -> AdapterState:
        prompt = sample.get("prompt", "")
        image = _ensure_pil_image(sample.get("image"))

        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    [{"type": "image"}] if image is not None else []
                )
                + [{"type": "text", "text": prompt}],
            }
        ]

        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        processor_inputs = self.processor(
            text=[chat_text],
            images=[image] if image is not None else None,
            return_tensors="pt",
        )

        processor_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in processor_inputs.items()}
        input_ids = processor_inputs.get("input_ids")
        attention_mask = processor_inputs.get("attention_mask")
        pixel_values = processor_inputs.get("pixel_values")
        image_sizes = processor_inputs.get("image_sizes")
        image_grid_thw = processor_inputs.get("image_grid_thw")
        image_token_index = processor_inputs.get("image_token_index")

        return AdapterState(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            image_grid_thw=image_grid_thw,
            image_token_index=image_token_index,
            step=0,
            past_key_values=None,
            metadata={"prompt": prompt},
        )


class MiniCPMVAdapter(HuggingFaceVLAdapter):
    def __init__(
        self,
        model_name: str = "open-mmlab/MiniCPM-V",
        device: str = "cuda",
        torch_dtype: Optional[str] = "float16",
        revision: Optional[str] = None,
        attn_implementation: Optional[str] = None,
    ) -> None:
        model_loader = None
        try:
            from transformers import MiniCPMVForConditionalGeneration

            model_loader = MiniCPMVForConditionalGeneration.from_pretrained
        except ImportError:
            # fall back to AutoModelForCausalLM via base class
            model_loader = None
        super().__init__(
            model_name=model_name,
            device=device,
            torch_dtype=torch_dtype,
            revision=revision,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
            model_loader=model_loader,
        )
