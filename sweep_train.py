#!/usr/bin/env python
"""
Wrapper script for W&B sweeps.
This allows wandb agent to properly run the Hydra-based runner.
Converts W&B's --arg=value format to Hydra's arg=value format.
"""
import sys
import os

# Get the directory where this script is located (project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Change to project root directory
os.chdir(SCRIPT_DIR)

# Add src to path
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'src'))

def convert_wandb_args_to_hydra():
    """Convert W&B's --arg=value format to Hydra's arg=value format."""
    hydra_args = []
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            # Remove leading --
            hydra_args.append(arg[2:])
        else:
            hydra_args.append(arg)
    return hydra_args

if __name__ == "__main__":
    # Convert arguments
    hydra_args = convert_wandb_args_to_hydra()

    # Replace sys.argv with converted args
    sys.argv = [sys.argv[0]] + hydra_args

    # Print debug info
    print(f"Working directory: {os.getcwd()}")
    print(f"Arguments: {sys.argv}")

    # Import and run
    from svlm.runner import main
    main()
