#!/bin/bash

# Experiment 10: Confusion Matrices
# Run confusion matrix experiment comparing 4 systems

cd "$(dirname "$0")/.." || exit 1

# Export API key if not already set
if [ -z "$OPENROUTER_API_KEY" ]; then
    if [ -f .env ]; then
        export $(grep -v '^#' .env | xargs)
    fi
fi

# Run experiment in retrieval-only mode to save API costs
echo "Running Experiment 10: Confusion Matrices (Retrieval-Only Mode)"
echo "================================================================"
python3 experiments/exp10_confusion_matrices.py \
    --num-questions 30 \
    --epsilon 1.0 \
    --retrieval-only

echo ""
echo "Experiment 10 complete!"
echo "Results saved to: experiments/results/exp10_confusion_matrices.png"
echo "Report saved to: experiments/results/exp10_confusion_matrices_report.txt"
