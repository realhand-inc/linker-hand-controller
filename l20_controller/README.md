# L20 Hand Controller Package

This package contains the control logic for the L20 Linker Hand.

## Structure

- `controller.py`: Main control loop and orchestration `L20Controller` class.
- `kinematics.py`: Hand-specific geometric calculations and joint angle logic.
- `math_utils.py`: Generic 3D vector math helper functions.
- `network.py`: ZMQ communication and message parsing.
- `main.py`: Entry point for running the controller directly.

## Usage

You can run the controller from the project root:

```bash
python3 l20_controller.py
```

Or run the module directly:

```bash
python3 -m l20_controller.main
```
