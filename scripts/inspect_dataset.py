"""Inspect a LeRobot dataset's features and column types to diagnose loading issues.

Usage:
    uv run scripts/inspect_dataset.py --config-name pi0.5_deft_legacy
    uv run scripts/inspect_dataset.py --repo-id your_hf_username/your_dataset
"""

import dataclasses
from PIL import Image as PILImage
import tyro

import openpi.training.config as _config


@dataclasses.dataclass
class Args:
    config_name: str | None = None
    repo_id: str | None = None


def main(args: Args) -> None:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    if args.repo_id:
        repo_id = args.repo_id
    elif args.config_name:
        all_configs = {c.name: c for c in _config.all_configs()}
        if args.config_name not in all_configs:
            raise ValueError(f"Unknown config: {args.config_name!r}. Available: {list(all_configs)}")
        train_config = all_configs[args.config_name]
        # Create a minimal DataConfig just to get the repo_id.
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        repo_id = data_config.repo_id
    else:
        raise ValueError("Provide either --config-name or --repo-id.")

    print(f"\n=== Inspecting dataset: {repo_id} ===\n")

    meta = LeRobotDatasetMetadata(repo_id)
    print("--- meta.features (from info.json) ---")
    for key, feat in meta.features.items():
        print(f"  {key}: dtype={feat.get('dtype')}, shape={feat.get('shape')}")

    print("\n--- Loading LeRobotDataset ---")
    ds = LeRobotDataset(repo_id)

    print("\n--- HuggingFace dataset features (parquet schema) ---")
    for key, feat in ds.hf_dataset.features.items():
        print(f"  {key}: {feat}")

    print("\n--- Runtime types of sample[0] ---")
    sample = ds.hf_dataset[0]
    problem_columns = []
    for key, value in sample.items():
        type_name = type(value).__name__
        preview = str(value)[:60].replace("\n", " ")
        print(f"  {key}: type={type_name}, preview={preview!r}")
        if isinstance(value, dict):
            problem_columns.append(key)

    if problem_columns:
        print(f"\n!!! PROBLEM COLUMNS (contain dict values): {problem_columns}")
        print("    These columns cannot be converted to tensors by LeRobot's hf_transform_to_torch.")
        print("    Most likely the image columns are not typed as datasets.Image() in the parquet schema.")
    else:
        print("\nNo dict-valued columns found in sample[0].")
        print("The issue may be in a different sample — check if any image is None or malformed.")


if __name__ == "__main__":
    tyro.cli(Args, main=main)
