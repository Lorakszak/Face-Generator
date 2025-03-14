#!/bin/bash

# Exit on error
set -e

echo "Setting up virtual environment for face prompt generation project..."

# Check for Python
if command -v python3.10 &>/dev/null; then
    PYTHON_CMD=python3.10
    echo "Found Python 3.10 (recommended version)"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
    PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ "$PY_VERSION" == "3.10" ]]; then
        echo "Found Python 3.10 (recommended version)"
    else
        echo "Warning: Python 3.10 is the recommended version, but found Python $PY_VERSION"
        echo "Continuing with Python $PY_VERSION, but some features may not work as expected."
    fi
else
    echo "Error: Python 3 is not installed."
    echo "Please install Python (preferably version 3.10) and try again."
    exit 1
fi

echo "Using Python: $($PYTHON_CMD --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found."
fi

echo ""
echo "Setup complete! To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the project after activation, you can use:"
echo "  python generate_dataset.py"
echo "AFTER setting up the configuration parameters."
echo ""
echo "To deactivate the virtual environment when finished, run:"
echo "  deactivate"
