#!/bin/bash
# Helper script to run Experiment 6: System Comparison

cd "$(dirname "$0")/.." || exit 1

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Set OpenRouter API key if not already set
if [ -z "$OPENROUTER_API_KEY" ]; then
    export OPENROUTER_API_KEY="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"
    echo "✓ Set OPENROUTER_API_KEY from script"
else
    echo "✓ Using existing OPENROUTER_API_KEY from environment"
fi

# Run experiment
echo "Running Experiment 6: System Comparison..."
python experiments/exp6_system_comparison.py "$@"
