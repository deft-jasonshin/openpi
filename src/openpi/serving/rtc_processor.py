from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
import math

import torch
from torch import Tensor


class PrefixAttentionSchedule(str, Enum):
    """Supported RTC prefix-weight schedules."""

    ZEROS = "zeros"
    ONES = "ones"
    LINEAR = "linear"
    EXP = "exp"


@dataclass(slots=True)
class RTCConfig:
    """Configuration for Real-Time Chunking guidance."""

    enabled: bool = False
    prefix_attention_schedule: PrefixAttentionSchedule = PrefixAttentionSchedule.EXP
    max_guidance_weight: float = 5.0

    def __post_init__(self) -> None:
        if self.max_guidance_weight <= 0:
            raise ValueError(
                f"max_guidance_weight must be positive, got {self.max_guidance_weight}"
            )


class RTCProcessor:
    """Implements the RTC soft mask and IIGDM guidance correction."""

    def __init__(self, rtc_config: RTCConfig | None = None) -> None:
        self.rtc_config = rtc_config or RTCConfig()

    def get_prefix_weights(
        self,
        inference_delay: int,
        execution_horizon: int,
        action_horizon: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Build RTC prefix weights for a chunk of length ``action_horizon``.

        The returned weights follow Eq. 5 from the RTC paper:
        - indices ``[0, d)`` are frozen with weight 1
        - indices ``[d, H - s)`` decay according to the configured schedule
        - indices ``[H - s, H)`` are unconstrained with weight 0
        """

        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")
        if inference_delay < 0:
            raise ValueError(f"inference_delay must be non-negative, got {inference_delay}")
        if execution_horizon < 0:
            raise ValueError(
                f"execution_horizon must be non-negative, got {execution_horizon}"
            )

        overlap_end = max(action_horizon - execution_horizon, 0)
        start = min(inference_delay, overlap_end)
        dtype = dtype or torch.float32

        if self.rtc_config.prefix_attention_schedule == PrefixAttentionSchedule.ZEROS:
            weights = torch.zeros(action_horizon, dtype=dtype, device=device)
            weights[:start] = 1.0
            return weights

        if self.rtc_config.prefix_attention_schedule == PrefixAttentionSchedule.ONES:
            weights = torch.zeros(action_horizon, dtype=dtype, device=device)
            weights[:overlap_end] = 1.0
            return weights

        mid_weights = self._linear_weights(start, overlap_end, device=device, dtype=dtype)
        if self.rtc_config.prefix_attention_schedule == PrefixAttentionSchedule.EXP:
            mid_weights = mid_weights * torch.expm1(mid_weights) / (math.e - 1)

        weights = self._add_trailing_zeros(mid_weights, action_horizon, overlap_end)
        weights = self._add_leading_ones(weights, start, action_horizon)
        return weights

    def guided_denoise_step(
        self,
        x_t: Tensor,
        time: float | Tensor,
        base_denoise_fn: Callable[[Tensor], Tensor],
        prev_chunk: Tensor | None,
        *,
        weights: Tensor | None = None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
        max_guidance_weight: float | None = None,
    ) -> Tensor:
        """Apply one RTC-guided denoising step.

        Args:
            x_t: Current latent action chunk of shape ``(B, H, A)`` or ``(H, A)``.
            time: Current denoising time in the model's convention, where time
                decreases from 1 to 0 during sampling.
            base_denoise_fn: Callable that maps ``x_t`` to the base velocity field.
            prev_chunk: Leftover actions from the previous chunk. If ``None``,
                the base velocity is returned unchanged.
            weights: Optional precomputed prefix weights of shape ``(H,)`` or
                broadcastable to ``x_t``.
            inference_delay: Required when ``weights`` is not provided.
            execution_horizon: Required when ``weights`` is not provided.
            max_guidance_weight: Optional override for the config value.
        """

        if prev_chunk is None:
            return base_denoise_fn(x_t)

        squeezed = False
        if x_t.ndim == 2:
            x_t = x_t.unsqueeze(0)
            squeezed = True
        if prev_chunk.ndim == 2:
            prev_chunk = prev_chunk.unsqueeze(0)

        x_t = x_t.detach().clone().requires_grad_(True)
        prev_chunk = self._pad_prev_chunk(prev_chunk, x_t)

        if weights is None:
            if inference_delay is None or execution_horizon is None:
                raise ValueError(
                    "Either weights or both inference_delay and execution_horizon must be provided."
                )
            weights = self.get_prefix_weights(
                inference_delay,
                execution_horizon,
                x_t.shape[1],
                device=x_t.device,
                dtype=x_t.dtype,
            )
        else:
            weights = weights.to(device=x_t.device, dtype=x_t.dtype)

        weights = self._broadcast_weights(weights, x_t)
        time_tensor = self._broadcast_time(time, x_t)
        guidance_weight = self._broadcast_time(
            self._compute_guidance_weight(time, x_t, max_guidance_weight), x_t
        )

        with torch.enable_grad():
            v_t = base_denoise_fn(x_t)
            x1_hat = x_t - time_tensor * v_t
            err = (prev_chunk - x1_hat) * weights
            correction = torch.autograd.grad(
                outputs=x1_hat,
                inputs=x_t,
                grad_outputs=err,
                retain_graph=False,
                create_graph=False,
            )[0]

        guided_velocity = v_t - guidance_weight * correction
        if squeezed:
            guided_velocity = guided_velocity.squeeze(0)
        return guided_velocity

    def denoise_step(
        self,
        x_t: Tensor,
        prev_chunk: Tensor | None,
        inference_delay: int,
        time: float | Tensor,
        base_denoise_fn: Callable[[Tensor], Tensor],
        *,
        execution_horizon: int,
        max_guidance_weight: float | None = None,
    ) -> Tensor:
        """Compatibility wrapper mirroring the upstream RTC processor API."""

        weights = self.get_prefix_weights(
            inference_delay,
            execution_horizon,
            x_t.shape[-2],
            device=x_t.device,
            dtype=x_t.dtype,
        )
        return self.guided_denoise_step(
            x_t,
            time,
            base_denoise_fn,
            prev_chunk,
            weights=weights,
            max_guidance_weight=max_guidance_weight,
        )

    def _compute_guidance_weight(
        self,
        time: float | Tensor,
        reference: Tensor,
        max_guidance_weight: float | None,
    ) -> Tensor:
        max_weight = max_guidance_weight or self.rtc_config.max_guidance_weight
        max_weight_tensor = torch.as_tensor(max_weight, device=reference.device, dtype=reference.dtype)

        time_tensor = torch.as_tensor(time, device=reference.device, dtype=reference.dtype)
        tau = 1 - time_tensor
        squared_one_minus_tau = (1 - tau) ** 2
        inv_r2 = torch.nan_to_num(
            (squared_one_minus_tau + tau**2) / squared_one_minus_tau,
            nan=max_weight,
            posinf=max_weight,
        )
        coeff = torch.nan_to_num((1 - tau) / tau, nan=max_weight, posinf=max_weight)
        guidance_weight = torch.nan_to_num(coeff * inv_r2, nan=max_weight, posinf=max_weight)
        return torch.minimum(guidance_weight, max_weight_tensor)

    def _linear_weights(
        self,
        start: int,
        overlap_end: int,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype,
    ) -> Tensor:
        linspace_steps = overlap_end - start
        if linspace_steps <= 0:
            return torch.empty(0, dtype=dtype, device=device)
        return torch.linspace(
            1.0,
            0.0,
            linspace_steps + 2,
            dtype=dtype,
            device=device,
        )[1:-1]

    def _add_trailing_zeros(self, weights: Tensor, total: int, overlap_end: int) -> Tensor:
        zeros_len = max(total - overlap_end, 0)
        if zeros_len == 0:
            return weights
        return torch.cat(
            [weights, torch.zeros(zeros_len, dtype=weights.dtype, device=weights.device)]
        )

    def _add_leading_ones(self, weights: Tensor, start: int, total: int) -> Tensor:
        ones_len = min(start, total)
        if ones_len == 0:
            return weights
        return torch.cat(
            [torch.ones(ones_len, dtype=weights.dtype, device=weights.device), weights]
        )

    def _pad_prev_chunk(self, prev_chunk: Tensor, x_t: Tensor) -> Tensor:
        prev_chunk = prev_chunk.to(device=x_t.device, dtype=x_t.dtype)
        if prev_chunk.shape == x_t.shape:
            return prev_chunk

        batch_size, action_horizon, action_dim = x_t.shape
        padded = torch.zeros(
            batch_size,
            action_horizon,
            action_dim,
            device=x_t.device,
            dtype=x_t.dtype,
        )
        padded[:, : prev_chunk.shape[1], : prev_chunk.shape[2]] = prev_chunk[
            :, :action_horizon, :action_dim
        ]
        return padded

    def _broadcast_weights(self, weights: Tensor, x_t: Tensor) -> Tensor:
        if weights.ndim == 1:
            weights = weights.unsqueeze(0).unsqueeze(-1)
        elif weights.ndim == 2:
            weights = weights.unsqueeze(-1)
        return weights

    def _broadcast_time(self, time: float | Tensor, x_t: Tensor) -> Tensor:
        time_tensor = torch.as_tensor(time, device=x_t.device, dtype=x_t.dtype)
        while time_tensor.ndim < x_t.ndim:
            time_tensor = time_tensor.unsqueeze(-1)
        return time_tensor


__all__ = ["PrefixAttentionSchedule", "RTCConfig", "RTCProcessor"]
