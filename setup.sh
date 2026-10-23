#!/bin/bash
# DrBinati MedAssist - Automated Setup Script
# Sets up a Python virtual environment, installs dependencies, and
# prepares the project directories for local development.

set -e

echo "======================================================================"
echo "DrBinati MedAssist - Automated Setup"
echo "======================================================================"

# 1. Check Python version
echo ""
echo "[1/7] Checking Python version..."
python3 --version

# 2. Create virtual environment
echo ""
echo "[2/7] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Virtual environment created."
else
    echo "  Virtual environment already exists, skipping."
fi

# 3. Activate virtual environment
echo ""
echo "[3/7] Activating virtual environment..."
source venv/bin/activate

# 4. Install dependencies
echo ""
echo "[4/7] Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Download clinical NLP model
echo ""
echo "[5/7] Downloading clinical NLP model (scispacy)..."
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_md-0.5.3.tar.gz || \
    echo "  Warning: scispacy model download failed, continuing without it."

# 6. Set up environment configuration
echo ""
echo "[6/7] Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  Created .env from .env.example - please add your API keys."
else
    echo "  .env already exists, skipping."
fi

# 7. Create required data directories
echo ""
echo "[7/7] Creating required directories..."
mkdir -p data/roco
mkdir -p data/test_samples
mkdir -p data/chroma_db
mkdir -p logs

echo ""
echo "======================================================================"
echo "Setup complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your ANTHROPIC_API_KEY or OPENAI_API_KEY"
echo "  2. Activate the environment: source venv/bin/activate"
echo "  3. Launch the app: streamlit run ui/streamlit_app.py"
echo ""
