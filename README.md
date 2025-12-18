# L20 Hand Controller Standalone

This is a standalone project for controlling the LinkerHand L20 using a GUI. It includes real-time calibration and visualization features.

## Project Structure

- l20_controller_gui.py: The main GUI application.
- l20_controller.py: Core logic for joint angle calculations.
- LinkerHand/: The LinkerHand Python SDK.
- requirements.txt: Python dependencies.

## Prerequisites

- Python: 3.7 or higher
- Hardware: LinkerHand L20 robotic hand with CAN interface
- ZMQ Data Source: Hand tracking application publishing MediaPipe landmarks on tcp://localhost:5557

## Installation

### Quick Install (Recommended)

Run the automated installation script:
```bash
cd /home/richard/linker-hand-controller
./install.sh
```

This script will:
- Create a virtual environment in `./venv`
- Install all required dependencies
- Handle the `externally-managed-environment` issue automatically

### Manual Installation

If you prefer to install manually:

1. Navigate to the project directory:
   ```bash
   cd /home/richard/linker-hand-controller
   ```

2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   ```

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   **Note:** If you encounter an `externally-managed-environment` error when installing packages, you must use a virtual environment (steps 2-3) or install system-wide using `apt install python3-<package>`.

## Configuration

Edit LinkerHand/config/setting.yaml to match your setup.

## Running the Application

1. Start the CAN interface (Linux):
   ```bash
   sudo ip link set can0 up type can bitrate 1000000
   ```

2. Activate the virtual environment (if you created one):
   ```bash
   source venv/bin/activate
   ```

3. Run the GUI:
   ```bash
   python3 l20_controller_gui.py
   ```

   Or run the command-line version without GUI:
   ```bash
   python3 l20_controller.py
   ```

4. In the GUI:
   - Click **Start Controller** to begin
   - Use **Calibrate MIN** (open hand) and **Calibrate MAX** (closed hand) for better accuracy
   - The display shows real-time joint angles and motor values