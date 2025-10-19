#!/usr/bin/env python
"""
W&B Sweep Launcher for SVLM Hyperparameter Search

Usage:
    # Start a Bayesian sweep
    python sweep_launcher.py --sweep-config configs/sweep.yaml

    # Start a grid search
    python sweep_launcher.py --sweep-config configs/sweep_grid.yaml

    # Start with specific number of runs
    python sweep_launcher.py --sweep-config configs/sweep_random.yaml --count 50

    # Resume existing sweep
    python sweep_launcher.py --sweep-id your-entity/svlm-calibration/sweep-id --count 10
"""

import argparse
import os
import sys
import yaml
import wandb


def create_sweep(config_path: str, project: str = "svlm-calibration", entity: str = None):
    """Create a new W&B sweep."""
    with open(config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)

    # Ensure program path is correct
    sweep_config['program'] = 'src/svlm/runner.py'

    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
    print(f"Created sweep: {sweep_id}")
    print(f"View at: https://wandb.ai/{entity or 'your-entity'}/{project}/sweeps/{sweep_id}")
    return sweep_id


def run_agent(sweep_id: str, count: int = None):
    """Run a W&B sweep agent."""
    print(f"Starting sweep agent for: {sweep_id}")
    if count:
        print(f"Will run {count} experiments")
        wandb.agent(sweep_id, function=run_experiment, count=count)
    else:
        print("Will run indefinitely (Ctrl+C to stop)")
        wandb.agent(sweep_id, function=run_experiment)


def run_experiment():
    """
    Run a single experiment with W&B sweep parameters.
    This is called by wandb.agent().
    """
    # Import here to avoid hydra conflicts
    from src.svlm.runner import main

    # wandb.init() is already called by sweep agent
    # Get sweep parameters
    config = wandb.config

    # Build hydra overrides from wandb config
    overrides = []
    for key, value in config.items():
        # Convert wandb config keys to hydra format
        # e.g., "method.lambda_" -> "method.lambda_=0.5"
        overrides.append(f"{key}={value}")

    print(f"Running with overrides: {overrides}")

    # Note: Hydra doesn't support programmatic overrides easily
    # So we'll use sys.argv manipulation
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]] + overrides

    try:
        main()
    finally:
        sys.argv = original_argv


def main_cli():
    parser = argparse.ArgumentParser(description="Launch W&B sweep for SVLM hyperparameter search")
    parser.add_argument(
        "--sweep-config",
        type=str,
        help="Path to sweep configuration YAML file"
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        help="Existing sweep ID to resume (format: entity/project/sweep-id)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of runs to execute (default: unlimited)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="svlm-calibration",
        help="W&B project name"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity (username or team)"
    )

    args = parser.parse_args()

    if not args.sweep_config and not args.sweep_id:
        parser.error("Must provide either --sweep-config or --sweep-id")

    if args.sweep_config and args.sweep_id:
        parser.error("Cannot provide both --sweep-config and --sweep-id")

    # Create or resume sweep
    if args.sweep_config:
        if not os.path.exists(args.sweep_config):
            print(f"Error: Config file not found: {args.sweep_config}")
            sys.exit(1)

        sweep_id = create_sweep(args.sweep_config, args.project, args.entity)
        print(f"\nSweep created! Now run:")
        print(f"  python sweep_launcher.py --sweep-id {sweep_id} --count {args.count or 10}")
        print(f"\nOr to run agent immediately:")
        response = input("Start agent now? [y/N]: ")
        if response.lower() == 'y':
            run_agent(sweep_id, args.count)
    else:
        run_agent(args.sweep_id, args.count)


if __name__ == "__main__":
    main_cli()
