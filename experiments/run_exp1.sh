#!/bin/bash
# Simple script to run Experiment 1: Federated Retrieval Proof

cd "$(dirname "$0")/.." || exit 1

echo "Running Experiment 1: Federated Retrieval Proof"
echo "================================================"
echo ""

# Check if tabulate is installed
python -c "import tabulate" 2>/dev/null || {
    echo "Installing tabulate..."
    pip install tabulate
}

# Run the experiment
python experiments/exp1_federated_retrieval_proof.py "$@"
