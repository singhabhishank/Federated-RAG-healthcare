#!/bin/bash
# Simple script to run Experiment 3: Privacy-Utility Tradeoff

cd "$(dirname "$0")/.." || exit 1

echo "Running Experiment 3: Privacy-Utility Tradeoff"
echo "================================================"
echo ""

# Run the experiment
python experiments/exp3_privacy_utility_tradeoff.py "$@"
