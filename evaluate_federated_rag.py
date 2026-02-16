#!/usr/bin/env python3
"""
Wrapper script to run the evaluation from the root directory.
This script sets up the environment and calls the evaluation script in src/.
"""

import os
import sys
from pathlib import Path

# Set environment variable for Groq API key if not already set
if not os.getenv("GROQ_API_KEY"):
    groq_key = "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"
    os.environ["GROQ_API_KEY"] = groq_key

# Add src directory to path
SRC_PATH = str(Path(__file__).parent / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Change to the directory containing this script (root directory)
os.chdir(Path(__file__).parent)

# Import and run the actual evaluation script
# The evaluation script's main() function will handle argparse
if __name__ == "__main__":
    from src.evaluate_federated_rag import main
    main()

