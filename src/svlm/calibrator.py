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
    # Automatic scaling parameters
    auto_scale_lambda: bool = False
    target_logit_impact: float = 0.05  # Target 5% impact on logits
    max_lambda: float = 200.0
    min_lambda: float = 0.01

    def __post_init__(self) -> None:
        if not (0.0 <= self.beta < 1.0):
            raise ValueError("beta must be in [0, 1)")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.lambda_ < 0.0:
            raise ValueError("lambda must be non-negative.")
        if not 0.0 < self.target_logit_impact <= 1.0:
            raise ValueError("target_logit_impact must be in (0, 1].")
        if self.max_lambda <= self.min_lambda:
            raise ValueError("max_lambda must be greater than min_lambda.")
        self._pva_state: Optional[torch.Tensor] = None
        self._debug_info: Optional[Dict[str, Optional[torch.Tensor]]] = None

    def reset(self) -> None:
        self._pva_state = None
        self._debug_info = None

    def _compute_auto_lambda(self, logits: torch.Tensor, signal_magnitude: float) -> float:
        """
        Automatically compute lambda based on logit scale and signal magnitude.
        Uses adaptive scaling to avoid excessive clipping.

        Args:
            logits: Current logits tensor
            signal_magnitude: Average absolute signal value

        Returns:
            Automatically scaled lambda value
        """
        if not self.auto_scale_lambda:
            return self.lambda_

        # Compute logit statistics
        logit_norm = torch.linalg.vector_norm(logits).item()
        logit_std = torch.std(logits).item()
        logit_mean = torch.mean(torch.abs(logits)).item()

        # Use multiple scaling strategies
        strategies = {}

        # Strategy 1: Target absolute change (more conservative)
        target_absolute_change = self.target_logit_impact * logit_mean
        strategies["absolute"] = target_absolute_change / (signal_magnitude + 1e-8)

        # Strategy 2: Relative to logit variance (more adaptive)
        strategies["variance"] = (logit_std * self.target_logit_impact) / (signal_magnitude + 1e-8)

        # Strategy 3: Logarithmic scaling (prevents extreme values)
        log_scale_factor = torch.log10(torch.tensor(logit_norm + 1)).item()
        strategies["logarithmic"] = (log_scale_factor * 10) / (signal_magnitude + 1e-8)

        # Choose the most appropriate strategy based on signal strength
        if signal_magnitude < 0.001:  # Very weak signal - use aggressive scaling
            required_lambda = strategies["logarithmic"] * 5
        elif signal_magnitude < 0.01:  # Weak signal - use moderate scaling
            required_lambda = strategies["absolute"] * 2
        else:  # Strong signal - use conservative scaling
            required_lambda = strategies["variance"]

        # Apply adaptive bounds based on logit scale
        dynamic_max = min(self.max_lambda, logit_norm * 0.5)  # Max 50% of logit norm
        dynamic_min = max(self.min_lambda, logit_norm * 0.001)  # Min 0.1% of logit norm

        auto_lambda = max(dynamic_min, min(dynamic_max, required_lambda))

        # Store debug info
        if self._debug_info is None:
            self._debug_info = {}
        self._debug_info["auto_lambda"] = torch.tensor([auto_lambda], device=logits.device).detach().to("cpu")
        self._debug_info["logit_norm"] = torch.tensor([logit_norm], device=logits.device).detach().to("cpu")
        self._debug_info["signal_magnitude"] = torch.tensor([signal_magnitude], device=logits.device).detach().to("cpu")
        self._debug_info["strategy_absolute"] = torch.tensor([strategies["absolute"]], device=logits.device).detach().to("cpu")
        self._debug_info["strategy_variance"] = torch.tensor([strategies["variance"]], device=logits.device).detach().to("cpu")
        self._debug_info["strategy_logarithmic"] = torch.tensor([strategies["logarithmic"]], device=logits.device).detach().to("cpu")
        self._debug_info["dynamic_max"] = torch.tensor([dynamic_max], device=logits.device).detach().to("cpu")
        self._debug_info["clipped"] = torch.tensor([1.0 if required_lambda > dynamic_max else 0.0], device=logits.device).detach().to("cpu")

        return auto_lambda

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
            adjusted_lambda = self.lambda_
            if self.auto_scale_lambda:
                adjusted_lambda = self._compute_auto_lambda(logits, float(centered_signal.abs().mean().item()))
            return logits + adjusted_lambda * centered_signal

        token_signal = self._map_signal_to_vocab(signal, token_ids, logits)
        if token_signal is None:
            return logits
        vocab_indices, deltas = token_signal

        if deltas.numel() == 0:
            return logits

        signal_magnitude = float(deltas.abs().mean().item())

        # Use automatic lambda scaling if enabled
        if self.auto_scale_lambda:
            adjusted_lambda = self._compute_auto_lambda(logits, signal_magnitude)
        else:
            adjusted_lambda = self.lambda_

        # Apply simple logit adjustment
        adjusted = logits.clone()
        adjusted[vocab_indices] = logits[vocab_indices] + adjusted_lambda * deltas

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
