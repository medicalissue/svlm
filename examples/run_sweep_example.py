#!/usr/bin/env python
"""
Example: How to programmatically run W&B sweeps for SVLM

This shows how to create and run sweeps using Python API instead of CLI.
"""

import wandb

# Define sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'pope_f1',
        'goal': 'maximize'
    },
    'parameters': {
        'method.lambda_': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 1.0
        },
        'method.beta': {
            'distribution': 'uniform',
            'min': 0.7,
            'max': 0.95
        },
        'method.alpha': {
            'distribution': 'uniform',
            'min': 0.3,
            'max': 0.9
        },
        # Fixed parameters
        'data': {
            'value': 'pope_adversarial'
        },
        'method.use_erw': {
            'value': True
        },
        'method.use_pva': {
            'value': True
        },
        'method.use_ven': {
            'value': True
        }
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 3,
        'eta': 2
    }
}

# Create sweep
sweep_id = wandb.sweep(
    sweep_config,
    project='svlm-calibration'
)

print(f"Sweep created: {sweep_id}")
print(f"Run with: wandb agent {sweep_id}")

# Or run agent directly
# wandb.agent(sweep_id, function=train, count=10)
