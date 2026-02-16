# Experiments Directory

This directory contains experimental scripts to demonstrate and validate different aspects of the Privacy-Preserving Federated RAG system.

## Structure

```
experiments/
├── README.md                    # This file
├── configs/                     # Experiment-specific configurations
├── results/                     # Experiment results and outputs
├── scripts/                     # Helper scripts
├── logs/                        # Experiment logs
├── exp1_federated_retrieval_proof.py  # Experiment 1 script
├── exp2_retrieval_relevance_plot.py   # Experiment 2 script
├── exp3_privacy_utility_tradeoff.py   # Experiment 3 script
├── exp4_client_contribution_heatmap.py # Experiment 4 script
├── exp5_end_to_end_walkthrough.py     # Experiment 5 script
├── exp6_system_comparison.py          # Experiment 6 script
├── exp7_reviewer_metrics_comparison.py # Experiment 7 script
├── exp8_relevance_comparison.py       # Experiment 8 script (Figure A)
└── exp9_privacy_utility_scatter.py   # Experiment 9 script (Figure B)
```

## Experiment 1: Federated Retrieval Proof

**Goal**: Prove each client only searches its own private database.

## Experiment 2: Retrieval Relevance Plot

**Goal**: Show retrieval quality is consistently high across multiple questions.

## Experiment 3: Privacy-Utility Tradeoff

**Goal**: Show that Differential Privacy is applied and doesn't destroy usefulness.

## Experiment 4: Client Contribution Heatmap

**Goal**: Prove federated collaboration by visualizing client contributions across questions.

## Experiment 5: End-to-End Walkthrough Diagram

**Goal**: Create the easiest visual proof for a supervisor showing the complete pipeline.

## Experiment 6: System Comparison

**Goal**: Compare 4 different RAG system configurations to isolate the effect of federation and privacy.

## Experiment 7: Reviewer-Approved Metrics Comparison

**Goal**: Comprehensive comparison using reviewer-approved metrics (retrieval quality, privacy, federated behavior, system cost).

## Experiment 8: Relevance Comparison (Figure A - REQUIRED)

**Goal**: Show that privacy introduces minimal degradation in retrieval quality (Figure A - REQUIRED).

## Experiment 9: Privacy vs Utility Trade-off (Figure B - STRONG)

**Goal**: Show that our model achieves optimal privacy–utility balance (Figure B - STRONG).

**What it demonstrates**:
- Each client retrieves documents from its own private ChromaDB
- Individual client results are different (proving data isolation)
- Aggregated results combine and rerank documents from all clients
- Shows which clients contributed to the final answer

**Visual Output**:
- Table for Client 0 top-5 documents
- Table for Client 1 top-5 documents  
- Table for Client 2 top-5 documents
- Final aggregated top-15 list

**For each document shows**:
- Title
- Journal
- Year
- Similarity score
- Short snippet (from abstract)

### Running Experiment 1

```bash
# From project root
cd experiments
python exp1_federated_retrieval_proof.py

# With custom question
python exp1_federated_retrieval_proof.py --question "What are the symptoms of diabetes?"
```

### Running Experiment 2

```bash
# From project root
python experiments/exp2_retrieval_relevance_plot.py

# Or use the shell script
./experiments/run_exp2.sh

# With custom number of questions
python experiments/exp2_retrieval_relevance_plot.py --num-questions 30
```

### Running Experiment 3

```bash
# From project root
python experiments/exp3_privacy_utility_tradeoff.py

# Or use the shell script
./experiments/run_exp3.sh

# With custom number of questions
python experiments/exp3_privacy_utility_tradeoff.py --num-questions 30

# With custom epsilon values
python experiments/exp3_privacy_utility_tradeoff.py --epsilons 0.1 0.5 1.0 2.0
```

### Running Experiment 4

```bash
# From project root
python experiments/exp4_client_contribution_heatmap.py

# Or use the shell script
./experiments/run_exp4.sh

# With custom number of questions
python experiments/exp4_client_contribution_heatmap.py --num-questions 30
```

### Running Experiment 5

```bash
# From project root
python experiments/exp5_end_to_end_walkthrough.py

# Or use the shell script
./experiments/run_exp5.sh

# With custom question
python experiments/exp5_end_to_end_walkthrough.py --question "What are the symptoms of diabetes?"
```

### Running Experiment 6

```bash
# From project root
python experiments/exp6_system_comparison.py

# Or use the shell script
./experiments/run_exp6.sh

# With custom number of questions
python experiments/exp6_system_comparison.py --num-questions 20

# With custom epsilon value
python experiments/exp6_system_comparison.py --epsilon 0.5
```

### Running Experiment 7

```bash
# From project root
python experiments/exp7_reviewer_metrics_comparison.py

# Or use the helper script
./experiments/run_exp7.sh

# With custom number of questions
python experiments/exp7_reviewer_metrics_comparison.py --num-questions 30

# With custom epsilon value
python experiments/exp7_reviewer_metrics_comparison.py --epsilon 0.5
```

### Running Experiment 8

```bash
# From project root
python experiments/exp8_relevance_comparison.py

# Or use the helper script
./experiments/run_exp8.sh

# With custom number of questions
python experiments/exp8_relevance_comparison.py --num-questions 50

# With custom epsilon value
python experiments/exp8_relevance_comparison.py --epsilon 0.5
```

### Running Experiment 9

```bash
# From project root
python experiments/exp9_privacy_utility_scatter.py

# Or use the helper script
./experiments/run_exp9.sh

# With custom epsilon values
python experiments/exp9_privacy_utility_scatter.py --epsilon-values 0.1 0.5 1.0 1.5 2.0

# With custom number of questions
python experiments/exp9_privacy_utility_scatter.py --num-questions 30
```

### Expected Output

The experiment will:
1. Create 3 federated clients with private vector databases
2. Retrieve top-5 documents from each client separately
3. Display formatted tables showing individual client results
4. Display aggregated top-15 results
5. Save results to `results/exp1_federated_retrieval_proof.txt`

### What "Perfect" Results Look Like

✅ **Each client list is different** - proves data isolation
✅ **The aggregated list is a merged/reranked combination** - shows federated aggregation works
✅ **Clients contributed = {0, 1, 2}** - all clients participate

### Requirements

- **Experiment 1**: No special requirements (uses custom table formatter)
- **Experiment 2**: Requires `matplotlib` and `numpy` (usually already installed)
