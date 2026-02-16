#!/bin/bash
# Simple script to run Experiment 2: Retrieval Relevance Plot

cd "$(dirname "$0")/.." || exit 1

echo "Running Experiment 2: Retrieval Relevance Plot"
echo "================================================"
echo ""

# Run the experiment
python experiments/exp2_retrieval_relevance_plot.py "$@"
