#!/bin/bash
# Simple script to run Experiment 5: End-to-End Walkthrough Diagram

cd "$(dirname "$0")/.." || exit 1

echo "Running Experiment 5: End-to-End Walkthrough Diagram"
echo "===================================================="
echo ""

# Run the experiment
python experiments/exp5_end_to_end_walkthrough.py "$@"
