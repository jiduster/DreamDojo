#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir

from cosmos_predict2.action_conditioned import inference
from cosmos_predict2.action_conditioned_config import (
    ActionConditionedInferenceArguments,
    ActionConditionedSetupArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run action-conditioned DreamDojo evaluation from a local checkpoint without HF/latest_checkpoint.txt."
    )
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default="2B/robot/action-cond")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument(
        "--config-file",
        type=str,
        default="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
    )
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--data-split", type=str, default="test")
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--deterministic-uniform-sampling", action="store_true")
    parser.add_argument("--single-base-index", action="store_true")
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_environment()
    try:
        init_output_dir(args.output_dir, profile=args.profile)

        setup_args = ActionConditionedSetupArguments(
            output_dir=args.output_dir,
            model=args.model,
            checkpoint_path=args.checkpoint_path,
            experiment=args.experiment,
            config_file=args.config_file,
            context_parallel_size=args.context_parallel_size,
            checkpoints_dir="",
            save_dir=args.save_dir,
            num_frames=args.num_frames,
            num_samples=args.num_samples,
            dataset_path=args.dataset_path,
            data_split=args.data_split,
            single_base_index=args.single_base_index,
            deterministic_uniform_sampling=args.deterministic_uniform_sampling,
            infinite=False,
            checkpoint_interval=2000,
        )
        inference_args = ActionConditionedInferenceArguments
        inference(setup_args, inference_args, Path(args.checkpoint_path))
    finally:
        cleanup_environment()


if __name__ == "__main__":
    main()
