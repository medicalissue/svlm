#!/bin/bash
# Create a new W&B sweep

SWEEP_TYPE=${1:-"bayes"}

case $SWEEP_TYPE in
    "bayes")
        CONFIG="configs/sweep.yaml"
        ;;
    "grid")
        CONFIG="configs/sweep_grid.yaml"
        ;;
    "random")
        CONFIG="configs/sweep_random.yaml"
        ;;
    *)
        echo "Usage: $0 [bayes|grid|random]"
        exit 1
        ;;
esac

echo "Creating $SWEEP_TYPE sweep from $CONFIG"
echo ""

# Create sweep and capture full output
OUTPUT=$(wandb sweep $CONFIG 2>&1)
echo "$OUTPUT"

# Extract sweep ID from output
SWEEP_ID=$(echo "$OUTPUT" | grep -o "wandb agent.*" | awk '{print $NF}')

if [ -n "$SWEEP_ID" ]; then
    echo ""
    echo "================================================================"
    echo "Sweep created successfully!"
    echo "================================================================"
    echo ""
    echo "To run the sweep, use:"
    echo "  ./run_agent.sh ${SWEEP_ID##*/} 20"
    echo ""
    echo "Or directly:"
    echo "  wandb agent --count 20 $SWEEP_ID"
    echo ""
fi
