import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_deft_example() -> dict:
    """Creates a random input example for the Deft policy."""
    return {
        "observation/left_joint_pos": np.random.rand(6),
        "observation/left_gripper_pos": np.random.rand(1),
        "observation/right_joint_pos": np.random.rand(6),
        "observation/right_gripper_pos": np.random.rand(1),
        "observation/base_state": np.random.rand(6),
        "observation/torso_state": np.random.rand(2),
        "observation/images/cam_high": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/images/cam_left_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/images/cam_right_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "action/left_joint_pos": np.random.rand(6),
        "action/left_gripper_pos": np.random.rand(1),
        "action/right_joint_pos": np.random.rand(6),
        "action/right_gripper_pos": np.random.rand(1),
        "action/base_cmd": np.random.rand(3),
        "action/lift_cmd": np.random.rand(1),
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
class DeftInputs(transforms.DataTransformFn):
    """
    Inputs for yam_ai_mobile.

    Expected inputs:
    - observation/{left,right}_joint_pos: [6]
    - observation/{left,right}_gripper_pos: [1]
    - observation/base_state: [6]
    - observation/torso_state: [2]
    - observation/images/{cam_high,cam_left_wrist,cam_right_wrist}: image
    - action/* split keys (during training) or "actions" pre-concatenated key
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.concatenate(
            [
                np.asarray(data["observation/left_joint_pos"]),
                np.atleast_1d(np.asarray(data["observation/left_gripper_pos"])),
                np.asarray(data["observation/right_joint_pos"]),
                np.atleast_1d(np.asarray(data["observation/right_gripper_pos"])),
                np.asarray(data["observation/base_state"]),
                np.asarray(data["observation/torso_state"]),
            ]
        )

        base_image = _parse_image(data["observation/images/cam_high"])
        left_wrist_image = _parse_image(data["observation/images/cam_left_wrist"])
        right_wrist_image = _parse_image(data["observation/images/cam_right_wrist"])

        inputs = {
            "state": state,
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

        if "action/left_joint_pos" in data:
            # Build 18-dim action vector:
            # left(6)+left_gripper(1)+right(6)+right_gripper(1)+base(3)+lift(1).
            inputs["actions"] = np.concatenate(
                [
                    np.asarray(data["action/left_joint_pos"]),
                    np.asarray(data["action/left_gripper_pos"]).reshape(-1, 1),
                    np.asarray(data["action/right_joint_pos"]),
                    np.asarray(data["action/right_gripper_pos"]).reshape(-1, 1),
                    np.asarray(data["action/base_cmd"]),
                    np.asarray(data["action/lift_cmd"]).reshape(-1, 1),
                ],
                axis=-1,
            )

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class DeftOutputs(transforms.DataTransformFn):
    """Outputs for yam_ai_mobile."""

    def __call__(self, data: dict) -> dict:
        # Keep only real robot action dimensions.
        return {"actions": np.asarray(data["actions"][:, :18])}