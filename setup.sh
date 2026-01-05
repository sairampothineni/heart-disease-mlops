#!/bin/bash

set -e  # Stop on first error

echo "🔹 Creating virtual environment..."
python -m venv venv

echo "🔹 Activating virtual environment..."
source venv/Scripts/activate

echo "🔹 Upgrading pip..."
python -m pip install --upgrade pip

echo "🔹 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "🔹 Running unit tests to verify reproducibility..."
pytest -v

echo "✅ Environment setup and reproducibility check completed successfully!"
