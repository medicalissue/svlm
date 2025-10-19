#!/bin/bash
# Run W&B sweep agent for existing sweep

if [ $# -lt 1 ]; then
    echo "Usage: $0 SWEEP_ID [COUNT]"
    echo "Example: $0 41doen3f 20"
    echo ""
    echo "Your existing sweep: 41doen3f"
    echo "View at: https://wandb.ai/medicalissues/svlm-calibration/sweeps/41doen3f"
    exit 1
fi

SWEEP_ID=$1
COUNT=${2:-10}

echo "Starting W&B agent for sweep: $SWEEP_ID"
echo "Will run $COUNT experiments"
echo "Press Ctrl+C to stop early"
echo ""

# Set working directory to project root
cd "$(dirname "$0")"

# Run agent with full sweep path
wandb agent --count $COUNT medicalissues/svlm-calibration/$SWEEP_ID
