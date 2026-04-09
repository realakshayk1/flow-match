#!/usr/bin/env bash
# Flow-match environment setup for Mac/Linux (pip + venv)
# Run once from the project root: bash setup.sh

set -e

echo "=== Creating virtual environment ==="
python3 -m venv .venv

echo "=== Activating venv ==="
source .venv/bin/activate

echo "=== Installing PyTorch (CPU) ==="
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

echo "=== Installing all other dependencies ==="
pip install -r requirements.txt

echo "=== Installing project as editable package ==="
pip install -e .

echo ""
echo "=== Done! ==="
echo "To activate in a new terminal, run:"
echo "    source .venv/bin/activate"
