#!/bin/bash
# Installation script for L20 Hand Controller
# This script sets up a virtual environment and installs all dependencies

set -e  # Exit on error

echo "========================================="
echo "L20 Hand Controller - Installation"
echo "========================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed"
    echo "Please install Python 3.7 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "Found Python version: $PYTHON_VERSION"
echo ""

# Create virtual environment if it doesn't exist
if [ -d "venv" ]; then
    echo "Virtual environment already exists at ./venv"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf venv
        echo "Creating new virtual environment..."
        python3 -m venv venv
    fi
else
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "========================================="
echo "Installation complete!"
echo "========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the GUI application:"
echo "  python3 l20_controller_gui.py"
echo ""
echo "To run the command-line version:"
echo "  python3 l20_controller.py"
echo ""
