import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_deft_legacy_example() -> dict:
    """Creates a random input example for the legacy Deft policy."""
    return {
        "observation/state": np.random.rand(68),
        "observation/images/cam_high": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/images/cam_left_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/images/cam_right_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "actions": np.random.rand(18),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DeftLegacyInputs(transforms.DataTransformFn):
    """
    Inputs for legacy yam_ai_mobile schema.

    Expected legacy inputs:
    - observation/state: [68] or [50]
    - observation/images/{cam_high,cam_left_wrist,cam_right_wrist}: image
    - actions: [action_horizon, 18] (training only)
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"])
        action_dim = 18
        # Keep only joint positions + base + torso from packed legacy state.
        # Supports both known variants:
        # - 68-dim: includes EE pos/orientation between arm torques and right-arm positions.
        # - 50-dim: no EE pos/orientation fields.
        if state.shape[-1] == 68:
            state = np.concatenate(
                [
                    state[..., 0:7],
                    state[..., 29:36],
                    state[..., 58:64],
                    state[..., 64:66],
                ],
                axis=-1,
            )
        elif state.shape[-1] == 50:
            state = np.concatenate(
                [
                    state[..., 0:7],
                    state[..., 21:28],
                    state[..., 42:48],
                    state[..., 48:50],
                ],
                axis=-1,
            )
        elif state.shape[-1] == 86:
            action_dim = 14
            state = np.concatenate(
                [
                    state[..., 0:7],
                    state[..., 36:43],
                ],
                axis=-1,
            )
        else:
            raise ValueError(
                f"Unsupported legacy observation/state dim: {state.shape[-1]}. "
                "Expected 68, 50, or 86."
            )

        base_image = _parse_image(data["observation/images/cam_high"])
        left_wrist_image = _parse_image(data["observation/images/cam_left_wrist"])
        right_wrist_image = _parse_image(data["observation/images/cam_right_wrist"])

        inputs = {
            "state": state,
            "legacy_action_dim": np.asarray(action_dim, dtype=np.int32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class DeftLegacyOutputs(transforms.DataTransformFn):
    """Outputs for legacy yam_ai_mobile schema."""

    def __call__(self, data: dict) -> dict:
        action_dim = int(np.asarray(data.get("legacy_action_dim", 18)).reshape(-1)[0])
        return {"actions": np.asarray(data["actions"][:, :action_dim])}
