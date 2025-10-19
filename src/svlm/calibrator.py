from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .attention import (
    AttentionProjections,
    compute_language_evidence,
    compute_visual_grounding,
    compute_visual_persistence,
    compute_visual_ratio,
    normalize_signal,
)


@dataclass
class LogitCalibrator:
    """
    Applies post-hoc logit calibration using ERW, PVA, VEN signals.
    """

    use_erw: bool = True
    use_pva: bool = False
    use_ven: bool = False
    lambda_: float = 0.3
    beta: float = 0.9
    alpha: float = 0.6
    ven_eps: float = 1e-5

    def __post_init__(self) -> None:
        if not (0.0 <= self.beta < 1.0):
            raise ValueError("beta must be in [0, 1)")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.lambda_ < 0.0:
            raise ValueError("lambda must be non-negative.")
        self._pva_state: Optional[torch.Tensor] = None
        self._debug_info: Optional[Dict[str, Optional[torch.Tensor]]] = None

    def reset(self) -> None:
        self._pva_state = None
        self._debug_info = None

    def _prepare(self, attn: AttentionProjections) -> AttentionProjections:
        attn.validate()
        return attn

    def adjust_logits(
        self,
        logits: torch.Tensor,
        attn: AttentionProjections,
        token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: Tensor [N_t] of pre-softmax logits for current step.
            attn: Attention projections for current step.
        """
        attn = self._prepare(attn)
        signal, debug = self._compute_signal(attn)
        if debug is not None:
            self._debug_info = {
                key: value.detach().to("cpu") if value is not None else None
                for key, value in debug.items()
            }
        if signal is None:
            return logits
        if token_ids is None:
            centered_signal, _ = normalize_signal(signal)
            return logits + self.lambda_ * centered_signal

        token_signal = self._map_signal_to_vocab(signal, token_ids, logits)
        if token_signal is None:
            return logits
        vocab_indices, deltas = token_signal
        adjusted = logits.clone()
        adjusted[vocab_indices] = logits[vocab_indices] + self.lambda_ * deltas
        return adjusted

    def get_last_signal_summary(self) -> Optional[Dict[str, Optional[Dict[str, float]]]]:
        if self._debug_info is None:
            return None
        summary: Dict[str, Optional[Dict[str, float]]] = {}
        for key, tensor in self._debug_info.items():
            if tensor is None:
                summary[key] = None
                continue
            if tensor.numel() == 0:
                summary[key] = {"mean": 0.0, "min": 0.0, "max": 0.0}
                continue
            data = tensor.detach().float()
            summary[key] = {
                "mean": float(data.mean().item()),
                "min": float(data.min().item()),
                "max": float(data.max().item()),
            }
        return summary

    def _compute_signal(
        self,
        attn: AttentionProjections,
    ) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, Optional[torch.Tensor]]]]:
        if not (self.use_erw or self.use_pva or self.use_ven):
            return None, None

        grounding = compute_visual_grounding(attn.cross_modal)
        debug: Dict[str, Optional[torch.Tensor]] = {"erw": grounding}

        pva_signal = None
        if self.use_pva:
            if self._pva_state is not None and self._pva_state.shape != grounding.shape:
                self._pva_state = None
            self._pva_state = compute_visual_persistence(grounding, self._pva_state, beta=self.beta)
            pva_signal = self._pva_state
        debug["pva"] = pva_signal

        ven_signal = None
        if self.use_ven:
            if attn.self_attn is None:
                raise ValueError("VEN requires self-attention projections.")
            language = compute_language_evidence(attn.self_attn)
            ven_signal = compute_visual_ratio(grounding, language, eps=self.ven_eps)
        debug["ven"] = ven_signal

        if self.use_pva and self.use_ven:
            combined = self.alpha * pva_signal + (1.0 - self.alpha) * ven_signal
            debug["combined"] = combined
            return combined, debug
        if self.use_pva:
            debug["combined"] = pva_signal
            return pva_signal, debug
        if self.use_ven:
            debug["combined"] = ven_signal
            return ven_signal, debug
        debug["combined"] = grounding
        return grounding, debug

    def _map_signal_to_vocab(
        self,
        signal: torch.Tensor,
        token_ids: torch.Tensor,
        logits: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if token_ids.dim() > 1:
            token_ids = token_ids.view(-1)
        token_ids = token_ids.to(torch.long).tolist()
        if not token_ids:
            return None
        device = logits.device
        dtype = logits.dtype
        signal = signal.to(device=device, dtype=dtype)

        values: Dict[int, List[torch.Tensor]] = {}
        for tid, s in zip(token_ids, signal):
            tid = int(tid)
            if tid < 0 or tid >= logits.shape[0]:
                continue
            values.setdefault(tid, []).append(s)

        if not values:
            return None

        vocab_indices = torch.tensor(list(values.keys()), device=device, dtype=torch.long)
        aggregated = torch.stack(
            [torch.stack(tensors).mean() for tensors in values.values()],
            dim=0,
        )
        centered, _ = normalize_signal(aggregated)
        return vocab_indices, centered
