"""Prepare and register paper review datasets for training."""

import argparse
import json
from pathlib import Path
from rllm.data.dataset import DatasetRegistry


def load_paper_tasks(data_path: str, max_samples: int = None):
    """Load paper review tasks from directory or file.

    Args:
        data_path: Path to data directory or JSON file
        max_samples: Maximum number of samples to load

    Returns:
        List of task dictionaries with paper_id, paper_content,
        human_avg_score, and clustered_points
    """
    data_path = Path(data_path)
    tasks = []

    if data_path.is_file():
        # Single JSON file
        with open(data_path) as f:
            data = json.load(f)
            tasks = data if isinstance(data, list) else [data]
    elif data_path.is_dir():
        # Directory of JSON files
        for json_file in sorted(data_path.glob("**/*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Extract required fields
                clustered_points = (
                    data.get("clustered_points_gpt-5mini")
                    or data.get("clustered_points")
                )
                if not clustered_points:
                    raise ValueError(f"No clustered_points found for {json_file.stem}")

                task = {
                    "paper_id": data.get("id", json_file.stem),
                    "paper_content": data.get("markdown", {}).get("content", ""),
                    "human_avg_score": data.get("scores", {}).get("rating_avg"),
                    "clustered_points": json.dumps(clustered_points),
                }

                # Prepend title if needed
                title = data.get("title", "")
                task["paper_content"] = f"# {title}\n\n{task['paper_content']}"

                # Only add if we have content and a valid human score
                human_score = task["human_avg_score"]
                if task["paper_content"] and human_score is not None and human_score != "null":
                    task["human_avg_score"] = float(human_score)
                    tasks.append(task)
                else:
                    if not human_score or human_score == "null":
                        print(f"Skipping {json_file.stem}: no human score")
                    elif not task["paper_content"]:
                        print(f"Skipping {json_file.stem}: no paper content")

            except Exception as e:
                print(f"Skipping {json_file}: {e}")

            if max_samples and len(tasks) >= max_samples:
                break
    else:
        raise ValueError(f"Data path not found: {data_path}")

    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and register paper review datasets"
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to paper data directory or JSON file"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.9,
        help="Fraction of data to use for training (default: 0.9)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to load (default: all)"
    )
    args = parser.parse_args()

    # Load tasks
    print(f"Loading tasks from {args.data_path}...")
    tasks = load_paper_tasks(args.data_path, args.max_samples)
    print(f"Loaded {len(tasks)} tasks")

    if not tasks:
        print("ERROR: No tasks loaded!")
        return

    # Split into train/val
    split_idx = int(len(tasks) * args.train_split)
    train_tasks = tasks[:split_idx]
    val_tasks = tasks[split_idx:]

    # Register datasets
    DatasetRegistry.register_dataset("paper_reviews", train_tasks, "train")
    DatasetRegistry.register_dataset("paper_reviews", val_tasks, "val")

    print(f"✓ Registered datasets:")
    print(f"  - train: {len(train_tasks)} samples")
    print(f"  - val: {len(val_tasks)} samples")
    print(f"\nDataset ready for training!")


if __name__ == "__main__":
    main()
