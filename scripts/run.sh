#!/bin/bash
# One-click run script for Transformer assignment
set -e

echo "=== Setting up environment ==="
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

echo "=== Starting Tiny Shakespeare convergence experiment ==="
python convergence_experiment.py --epochs 20 --batch_size 64

echo "=== Experiment complete! Results saved to experiments/ ==="
