from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Protocol

import torch

from .attention import AttentionProjections


@dataclass
class StepOutput:
    logits: torch.Tensor
    attention: AttentionProjections
    past_key_values: Any


@dataclass
class AdapterState:
    input_ids: torch.Tensor
    step: int = 0
    past_key_values: Any = None


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
            step=state.step + 1,
            past_key_values=None,
        )

    def decode(self, token_ids: Iterable[int]) -> str:
        return " ".join(str(tid) for tid in token_ids)

    @property
    def eos_token_id(self) -> Optional[int]:
        return None
