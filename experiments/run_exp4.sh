#!/bin/bash
# Simple script to run Experiment 4: Client Contribution Heatmap

cd "$(dirname "$0")/.." || exit 1

echo "Running Experiment 4: Client Contribution Heatmap"
echo "=================================================="
echo ""

# Run the experiment
python experiments/exp4_client_contribution_heatmap.py "$@"
