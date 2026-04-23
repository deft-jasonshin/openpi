import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DeftLatestInputs(transforms.DataTransformFn):
    """
    Inputs for yam_ai_mobile (v2.1 schema).

    Expected inputs:
    - observation/{left,right}_joint_pos: [6]
    - observation/{left,right}_gripper_pos: [1]
    - observation/base_state: [6]
    - observation/torso_state: [2]
    - observation/images/{cam_high,cam_left_wrist,cam_right_wrist}: video frames
    - action/left_joint_pos: [6]
    - action/left_gripper_pos: [1]
    - action/right_joint_pos: [6]
    - action/right_gripper_pos: [1]
    - action/base_target_state: [6]
    - action/lift_cmd: [1]
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

        actions = np.concatenate(
            [
                np.asarray(data["action/left_joint_pos"]),
                np.asarray(data["action/left_gripper_pos"]).reshape(-1, 1),
                np.asarray(data["action/right_joint_pos"]),
                np.asarray(data["action/right_gripper_pos"]).reshape(-1, 1),
                np.asarray(data["action/base_target_state"]),
                np.asarray(data["action/lift_cmd"]).reshape(-1, 1),
            ],
            axis=-1,
        )

        inputs: dict = {
            "state": state,
            "actions": actions,
        }

        if "observation/images/cam_high" in data:
            base_image = _parse_image(data["observation/images/cam_high"])
            left_wrist_image = _parse_image(data["observation/images/cam_left_wrist"])
            right_wrist_image = _parse_image(data["observation/images/cam_right_wrist"])
            inputs["image"] = {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            }
            inputs["image_mask"] = {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            }

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class DeftLatestOutputs(transforms.DataTransformFn):
    """Outputs for yam_ai_mobile (v2.1 schema)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :21])}
