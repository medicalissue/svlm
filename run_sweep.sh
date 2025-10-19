#!/bin/bash
# Quick sweep launcher for SVLM

# Configuration
SWEEP_TYPE=${1:-"bayes"}  # bayes, grid, or random
COUNT=${2:-10}

case $SWEEP_TYPE in
    "bayes")
        CONFIG="configs/sweep.yaml"
        echo "Starting Bayesian optimization sweep..."
        ;;
    "grid")
        CONFIG="configs/sweep_grid.yaml"
        echo "Starting grid search sweep..."
        ;;
    "random")
        CONFIG="configs/sweep_random.yaml"
        echo "Starting random search sweep..."
        ;;
    *)
        echo "Usage: $0 [bayes|grid|random] [count]"
        echo "Example: $0 bayes 20"
        exit 1
        ;;
esac

# Check if wandb is installed
if ! python -c "import wandb" 2>/dev/null; then
    echo "Error: wandb not installed. Run: pip install wandb"
    exit 1
fi

# Login to wandb if needed
if [ ! -f ~/.netrc ] || ! grep -q "api.wandb.ai" ~/.netrc; then
    echo "Please login to W&B:"
    wandb login
fi

# Create sweep and get ID
echo "Creating sweep with config: $CONFIG"
SWEEP_ID=$(python -c "
import yaml
import wandb

with open('$CONFIG') as f:
    config = yaml.safe_load(f)
config['program'] = 'src/svlm/runner.py'

sweep_id = wandb.sweep(config, project='svlm-calibration')
print(sweep_id)
")

echo "Sweep created: $SWEEP_ID"
echo "View at: https://wandb.ai/YOUR_USERNAME/svlm-calibration/sweeps/$SWEEP_ID"
echo ""
echo "Starting agent with $COUNT runs..."
echo "Press Ctrl+C to stop early"
echo ""

# Run agent
cd src/svlm
wandb agent --count $COUNT $SWEEP_ID
