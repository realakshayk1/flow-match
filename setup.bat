@echo off
:: Flow-match environment setup for Windows (pip + venv)
:: Run once from the project root: .\setup.bat

echo === Creating virtual environment ===
python -m venv .venv

echo === Activating venv ===
call .venv\Scripts\activate.bat

echo === Installing PyTorch (CPU) ===
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

echo === Installing all other dependencies ===
pip install -r requirements.txt

echo === Installing project as editable package ===
pip install -e .

echo.
echo === Done! ===
echo To activate in a new terminal, run:
echo     .venv\Scripts\activate.bat
