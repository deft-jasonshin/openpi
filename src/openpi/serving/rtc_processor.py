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

    prefix_attention_schedule: PrefixAttentionSchedule = PrefixAttentionSchedule.EXP
    max_guidance_weight: float = 5.0

    def __post_init__(self) -> None:
        if self.max_guidance_weight <= 0:
            raise ValueError(
                f"max_guidance_weight must be positive, got {self.max_guidance_weight}"
            )


class RTCProcessor:
    """Implements the RTC soft mask and IIGDM guidance correction.

    Instantiate once (e.g. in the model constructor) and reuse across calls.
    """

    def __init__(self, config: RTCConfig | None = None) -> None:
        self.config = config or RTCConfig()

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

        schedule = self.config.prefix_attention_schedule

        if schedule == PrefixAttentionSchedule.ZEROS:
            weights = torch.zeros(action_horizon, dtype=dtype, device=device)
            weights[:start] = 1.0
            return weights

        if schedule == PrefixAttentionSchedule.ONES:
            weights = torch.zeros(action_horizon, dtype=dtype, device=device)
            weights[:overlap_end] = 1.0
            return weights

        mid_weights = self._linear_weights(start, overlap_end, device=device, dtype=dtype)
        if schedule == PrefixAttentionSchedule.EXP:
            mid_weights = mid_weights * torch.expm1(mid_weights) / (math.e - 1)

        weights = self._assemble_weights(mid_weights, start, overlap_end, action_horizon)
        return weights

    def guided_denoise_step(
        self,
        x_t: Tensor,
        time: float | Tensor,
        base_denoise_fn: Callable[[Tensor], Tensor],
        prev_chunk: Tensor | None,
        inference_delay: int,
        execution_horizon: int,
        max_guidance_weight: float | None = None,
    ) -> Tensor:
        """Apply one RTC-guided denoising step (IIGDM correction).

        Args:
            x_t: Current latent action chunk ``(B, H, A)`` or ``(H, A)``.
            time: Denoising time (1 = pure noise, 0 = clean).
            base_denoise_fn: Maps ``x_t`` to the base velocity field.
            prev_chunk: Previous action chunk. ``None`` skips guidance.
            inference_delay: Steps already executed while computing this chunk.
            execution_horizon: Steps that will be executed from this chunk.
            max_guidance_weight: Clamp for the time-dependent guidance weight.
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

        weights = self.get_prefix_weights(
            inference_delay,
            execution_horizon,
            x_t.shape[1],
            device=x_t.device,
            dtype=x_t.dtype,
        )

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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_guidance_weight(
        self,
        time: float | Tensor,
        reference: Tensor,
        max_guidance_weight: float | None,
    ) -> Tensor:
        max_weight = max_guidance_weight or self.config.max_guidance_weight
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
        return torch.linspace(1.0, 0.0, linspace_steps + 2, dtype=dtype, device=device)[1:-1]

    def _assemble_weights(
        self, mid_weights: Tensor, start: int, overlap_end: int, total: int
    ) -> Tensor:
        """Concatenate leading ones, mid decay, and trailing zeros."""
        parts: list[Tensor] = []
        if start > 0:
            parts.append(torch.ones(min(start, total), dtype=mid_weights.dtype, device=mid_weights.device))
        parts.append(mid_weights)
        trailing = max(total - overlap_end, 0)
        if trailing > 0:
            parts.append(torch.zeros(trailing, dtype=mid_weights.dtype, device=mid_weights.device))
        return torch.cat(parts)

    def _pad_prev_chunk(self, prev_chunk: Tensor, x_t: Tensor) -> Tensor:
        prev_chunk = prev_chunk.to(device=x_t.device, dtype=x_t.dtype)
        if prev_chunk.shape == x_t.shape:
            return prev_chunk

        batch_size, action_horizon, action_dim = x_t.shape
        padded = torch.zeros(
            batch_size, action_horizon, action_dim,
            device=x_t.device, dtype=x_t.dtype,
        )
        padded[:, : prev_chunk.shape[1], : prev_chunk.shape[2]] = prev_chunk[
            :, :action_horizon, :action_dim
        ]
        return padded

    def _broadcast_weights(self, weights: Tensor, x_t: Tensor) -> Tensor:
        if weights.ndim == 1:
            return weights.unsqueeze(0).unsqueeze(-1)
        if weights.ndim == 2:
            return weights.unsqueeze(-1)
        return weights

    def _broadcast_time(self, time: float | Tensor, x_t: Tensor) -> Tensor:
        time_tensor = torch.as_tensor(time, device=x_t.device, dtype=x_t.dtype)
        while time_tensor.ndim < x_t.ndim:
            time_tensor = time_tensor.unsqueeze(-1)
        return time_tensor


__all__ = ["PrefixAttentionSchedule", "RTCConfig", "RTCProcessor"]
