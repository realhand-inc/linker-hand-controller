# L20 Hand Controller Package

This package contains the core logic for the L20 Linker Hand, used by the GUI application.

## Structure

- `kinematics.py`: Hand-specific geometric calculations and joint angle logic.
- `math_utils.py`: Generic 3D vector math helper functions.
- `network.py`: ZMQ communication utilities and message parsing.

## Usage

This package is intended to be used as a library by other applications (like `l20_controller_gui.py`).