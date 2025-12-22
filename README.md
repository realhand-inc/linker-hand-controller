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

1. Navigate to the project directory:
   

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
 

4. In the GUI:
   - Click **Start Controller** to begin
   - Use **Calibrate MIN** (open hand) and **Calibrate MAX** (closed hand) for better accuracy
   - The display shows real-time joint angles and motor values

## Data Flow Pipeline

```text
ZMQ SUB (tcp://localhost:5557)
            |
            v
     parse_landmarks()
            |
            v
calculate_all_joint_angles()
            |
            v
_get_raw_angle_for_motor()
            |
            v
CalibrationData.get_motor_value()
            |
            v
      _merge_poses()
            |
            v
LinkerHandApi.finger_move()
```

## Thumb Joint Calculations

The thumb kinematics system calculates three primary joint angles from MediaPipe hand landmarks:

### Thumb CMC Abduction

**Function**: `calculate_thumb_cmc_abduction()` in `l20_controller/kinematics.py:42-75`

**Purpose**: Measures thumb spread away from the palm (abduction/adduction movement).

**Thumb landmarks used**: Measures from **thumb CMC** (base) to **thumb TIP** (endpoint of thumb).

**Method**:
1. Calculate palm plane normal: `n = normalize(cross(middle_mcp - wrist, index_mcp - pinky_mcp))`
2. Define reference vector (palm width): `r = pinky_mcp → index_mcp`
3. Define thumb vector: `t = thumb_cmc → thumb_tip`
4. Project both vectors onto palm plane: `v_proj = v - (v·n)n`
5. Calculate angle: `θ = arccos(dot(r_proj_norm, t_proj_norm))`

**Range**: [0, π/2] radians (0 = adducted/against palm, π/2 = abducted/spread out)

### Thumb CMC Flexion

**Function**: `calculate_thumb_cmc_flexion()` in `l20_controller/kinematics.py:78-107`

**Purpose**: Measures forward/backward motion of the thumb base (flexion/extension).

**Thumb landmarks used**: Measures from **thumb CMC** (base) to **thumb MCP** (first knuckle).

**Method**:
1. Calculate abduction axis: `a = normalize(cross(index_mcp - wrist, middle_mcp - wrist))`
2. Define reference vector: `r = wrist → index_mcp`
3. Define thumb vector: `t = thumb_cmc → thumb_mcp`
4. Use `signed_angle_in_plane(r, t, a)` to get signed angle in plane perpendicular to abduction axis

**Range**: [0, π/2] radians (0 = extended, π/2 = flexed)

### Thumb MCP Flexion

**Function**: `calculate_thumb_mcp_flexion()` in `l20_controller/kinematics.py:131-152`

**Purpose**: Measures bending at the first thumb knuckle (metacarpophalangeal joint).

**Thumb landmarks used**: Calculates angle at **thumb MCP** (first knuckle) using **thumb CMC** (base) and **thumb IP** (second joint).

**Method**:
1. Define vector 1 (pointing to base): `v1 = thumb_mcp → thumb_cmc`
2. Define vector 2 (pointing to tip): `v2 = thumb_mcp → thumb_ip`
3. Calculate angle between bones: `θ = arccos(clamp(dot(normalize(v1), normalize(v2)), -1, 1))`

**Range**: [0, π] radians (0 = straight, π = fully flexed)

**Note**: All angles are calculated in radians and later converted to motor values (0-255) by the `angles_to_l20_pose()` function.

## Joint Data Structure

The system sends a 20-element pose array to the L20 robot hand hardware via CAN protocol. Each element is an 8-bit integer (0-255) controlling a specific motor.

### Motor Control Mapping

| Index | Motor Name | Description |
|-------|------------|-------------|
| **Base Flexion (0-4)** | | |
| 0 | `thumb_base` | Thumb base joint flexion |
| 1 | `index_base` | Index finger base joint flexion |
| 2 | `middle_base` | Middle finger base joint flexion |
| 3 | `ring_base` | Ring finger base joint flexion |
| 4 | `pinky_base` | Pinky finger base joint flexion |
| **Abduction/Spread (5-9)** | | |
| 5 | `thumb_abduction` | Thumb abduction/adduction |
| 6 | `index_spread` | Index finger spread |
| 7 | `middle_spread` | Middle finger spread |
| 8 | `ring_spread` | Ring finger spread |
| 9 | `pinky_spread` | Pinky finger spread |
| **Thumb Yaw + Reserved (10-14)** | | |
| 10 | `thumb_yaw` | Thumb yaw rotation to palm |
| 11-14 | `reserved_*` | Reserved (unused in current L20 implementation) |
| **Fingertip Flexion (15-19)** | | |
| 15 | `thumb_tip` | Thumb tip joint flexion |
| 16 | `index_tip` | Index fingertip flexion |
| 17 | `middle_tip` | Middle fingertip flexion |
| 18 | `ring_tip` | Ring fingertip flexion |
| 19 | `pinky_tip` | Pinky fingertip flexion |

**Value Range**: All motor values are integers in the range [0, 255].

## Closing a Stuck GUI Window

If you terminate the app but the window stays open, find the process and kill it:

```bash
pgrep -af l20_controller_gui.py
```

Then stop it:

```bash
kill -9 <PID>
```
