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
    contrastive_strength: float = 2.0
    contrastive_threshold: float = 0.5

    # Adversarial perturbation parameters
    adversarial_strength: float = 0.3
    adversarial_iterations: int = 3
    adversarial_step_size: float = 0.1

    # Adaptive temperature parameters
    temp_base: float = 1.0
    temp_range: float = 2.0
    temp_visual_boost: bool = True

    # Automatic scaling parameters
    auto_scale_lambda: bool = False
    target_logit_impact: float = 0.1  # Target 10% impact on logits
    max_lambda: float = 1000.0
    min_lambda: float = 0.1

    def __post_init__(self) -> None:
        if not (0.0 <= self.beta < 1.0):
            raise ValueError("beta must be in [0, 1)")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.lambda_ < 0.0:
            raise ValueError("lambda must be non-negative.")
        if self.candidate_topk <= 0:
            raise ValueError("candidate_topk must be positive.")
        if not 0.0 <= self.candidate_mix_cap <= 1.0:
            raise ValueError("candidate_mix_cap must be in [0, 1].")
        if not 0.0 <= self.residual_scale <= 1.0:
            raise ValueError("residual_scale must be in [0, 1].")
        if self.contrastive_strength <= 0.0:
            raise ValueError("contrastive_strength must be positive.")
        if not 0.0 <= self.contrastive_threshold <= 1.0:
            raise ValueError("contrastive_threshold must be in [0, 1].")
        if self.adversarial_strength < 0.0:
            raise ValueError("adversarial_strength must be non-negative.")
        if self.adversarial_iterations <= 0:
            raise ValueError("adversarial_iterations must be positive.")
        if self.temp_base <= 0.0:
            raise ValueError("temp_base must be positive.")
        if self.temp_range < 0.0:
            raise ValueError("temp_range must be non-negative.")
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

        def _record_debug(dynamic_lambda_value: float, mix_factor_value: float) -> None:
            if self._debug_info is None:
                self._debug_info = {}
            self._debug_info["dynamic_lambda"] = torch.tensor(
                [dynamic_lambda_value], device=logits.device
            ).detach().to("cpu")
            self._debug_info["mix_factor"] = torch.tensor(
                [mix_factor_value], device=logits.device
            ).detach().to("cpu")

        if token_ids is None:
            centered_signal, _ = normalize_signal(signal)
            dynamic_scale = 1.0
            if self.dynamic_lambda:
                dynamic_scale += torch.tanh(centered_signal.abs().mean())
            adjusted_lambda = self.lambda_ * dynamic_scale
            _record_debug(adjusted_lambda, 0.0)
            return logits + adjusted_lambda * centered_signal

        token_signal = self._map_signal_to_vocab(signal, token_ids, logits)
        if token_signal is None:
            return logits
        vocab_indices, deltas = token_signal

        if deltas.numel() == 0:
            _record_debug(self.lambda_, 0.0)
            return logits

        signal_strength_tensor = torch.tanh(deltas.abs().mean())
        signal_strength = float(signal_strength_tensor.item())
        signal_magnitude = float(deltas.abs().mean().item())

        # Use automatic lambda scaling if enabled
        if self.auto_scale_lambda:
            adjusted_lambda = self._compute_auto_lambda(logits, signal_magnitude)
        else:
            adjusted_lambda = self.lambda_
            if self.dynamic_lambda:
                adjusted_lambda *= 1.0 + signal_strength

        adjusted = logits.clone()
        mix_factor = 0.0

        if self.candidate_blending:
            top_k = min(self.candidate_topk, deltas.numel())
            scores = deltas.abs()
            _, top_positions = torch.topk(scores, top_k, largest=True)
            selected_indices = vocab_indices[top_positions]
            selected_deltas = deltas[top_positions]

            epsilon = torch.finfo(logits.dtype).eps if logits.dtype.is_floating_point else 1e-6
            base_probs = torch.softmax(logits[selected_indices], dim=0)
            signal_logits = selected_deltas * adjusted_lambda
            signal_probs = torch.softmax(signal_logits, dim=0)

            mix_factor = min(self.candidate_mix_cap, max(0.0, signal_strength))
            mixed_probs = (1.0 - mix_factor) * base_probs + mix_factor * signal_probs

            delta_logits = torch.log(mixed_probs + epsilon) - torch.log(base_probs + epsilon)
            adjusted[selected_indices] = logits[selected_indices] + delta_logits

            if deltas.numel() > top_k:
                mask = torch.ones_like(deltas, dtype=torch.bool)
                mask[top_positions] = False
                residual_indices = vocab_indices[mask]
                residual_deltas = deltas[mask]
                if residual_indices.numel() > 0:
                    adjusted[residual_indices] = logits[residual_indices] + adjusted_lambda * residual_deltas * self.residual_scale
        else:
            adjusted[vocab_indices] = logits[vocab_indices] + adjusted_lambda * deltas

        _record_debug(adjusted_lambda, mix_factor)

        # Apply extreme manipulation methods if enabled (only on mapped vocab positions)
        if self.use_contrastive:
            adjusted = self._apply_contrastive_sharpening(adjusted, vocab_indices, deltas)
        if self.use_adversarial:
            adjusted = self._apply_adversarial_perturbation(adjusted, vocab_indices, deltas)
        if self.use_adaptive_temp:
            adjusted = self._apply_adaptive_temperature(adjusted, vocab_indices, deltas)

        return adjusted

    def _apply_contrastive_sharpening(
        self,
        logits: torch.Tensor,
        vocab_indices: torch.Tensor,
        signal_deltas: torch.Tensor,
    ) -> torch.Tensor:
        """
        방안 1: Contrastive Logit Sharpening
        시각 정보 있는 토큰은 강화, 없는 토큰은 억제
        """
        if signal_deltas.numel() == 0:
            return logits

        # Signal에 따라 토큰을 두 그룹으로 분리
        signal_deltas_float = signal_deltas.float()
        threshold = torch.median(signal_deltas_float)

        # 시각적으로 grounded된 토큰 (양수 delta) 강화
        visual_mask = signal_deltas_float > threshold
        hallucination_mask = signal_deltas_float <= threshold

        adjusted = logits.clone()

        # 극단적 대비 (sharpening_factor를 파라미터화)
        if visual_mask.any():
            visual_indices = vocab_indices[visual_mask]
            adjusted[visual_indices] = adjusted[visual_indices] * self.contrastive_strength
        if hallucination_mask.any():
            hallucination_indices = vocab_indices[hallucination_mask]
            adjusted[hallucination_indices] = adjusted[hallucination_indices] / self.contrastive_strength

        return adjusted

    def _apply_adversarial_perturbation(
        self,
        logits: torch.Tensor,
        vocab_indices: torch.Tensor,
        signal_deltas: torch.Tensor,
    ) -> torch.Tensor:
        """
        방안 5: Adversarial Logit Perturbation
        반대 방향으로 교란 후 재조정
        """
        if signal_deltas.numel() == 0:
            return logits

        adjusted = logits.clone()
        signal_deltas_float = signal_deltas.float()
        median_signal = torch.median(signal_deltas_float)
        quantile_75 = torch.quantile(signal_deltas_float, 0.75)

        for _ in range(self.adversarial_iterations):
            # 현재 분포 (전체 vocab)
            probs = torch.softmax(adjusted, dim=0)

            # Signal과 반대로 가장 높은 확률 토큰 억제
            max_prob_idx = probs.argmax()

            # 이 토큰이 우리가 매핑한 토큰 중 하나인지 확인
            if max_prob_idx in vocab_indices:
                # 해당 토큰의 signal delta 찾기
                token_pos = (vocab_indices == max_prob_idx).nonzero(as_tuple=True)[0]
                if token_pos.numel() > 0:
                    token_signal = signal_deltas_float[token_pos[0]]
                    if token_signal < median_signal:
                        adjusted[max_prob_idx] -= self.adversarial_strength * 10.0  # 강력 억제

            # Signal 높은 토큰 강화
            high_signal_mask = signal_deltas_float > quantile_75
            if high_signal_mask.any():
                high_signal_indices = vocab_indices[high_signal_mask]
                adjusted[high_signal_indices] += self.adversarial_strength * 5.0

        return adjusted

    def _apply_adaptive_temperature(
        self,
        logits: torch.Tensor,
        vocab_indices: torch.Tensor,
        signal_deltas: torch.Tensor,
    ) -> torch.Tensor:
        """
        방안 3: Temperature Scaling per Token
        토큰별로 다른 temperature 적용
        """
        if signal_deltas.numel() == 0:
            return logits

        # Signal 강도에 따라 temperature 조절
        # 높은 signal = 낮은 temperature (sharper)
        # 낮은 signal = 높은 temperature (flatter)

        signal_deltas_float = signal_deltas.float()
        signal_min = signal_deltas_float.min()
        signal_max = signal_deltas_float.max()
        signal_range = signal_max - signal_min + 1e-8

        signal_normalized = (signal_deltas_float - signal_min) / signal_range

        # 토큰별 temperature 계산
        if self.temp_visual_boost:
            # 높은 signal = 낮은 temperature
            token_temps = self.temp_base + self.temp_range * (1.0 - signal_normalized)
        else:
            # 낮은 signal = 높은 temperature
            token_temps = self.temp_base + self.temp_range * signal_normalized

        # 매핑된 vocab 위치에만 적용
        adjusted = logits.clone()
        for i, vocab_idx in enumerate(vocab_indices):
            adjusted[vocab_idx] = adjusted[vocab_idx] / token_temps[i]

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
