#!/usr/bin/env python3
"""
Entry point for L20 Hand Controller.
"""
import sys
import os

# Ensure the current directory is in sys.path so we can import the package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from l20_controller.main import main

if __name__ == "__main__":
    main()