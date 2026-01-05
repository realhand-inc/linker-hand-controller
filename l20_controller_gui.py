#!/usr/bin/env python3
"""
L20 Hand Controller with GUI - Real-time Calibration and Visualization

Features:
- Live display of all joint angles
- Calibration Min/Max buttons for accurate range mapping
- Real-time angle remapping based on user's actual range of motion
- Visual feedback of joint states
"""
import json
import math
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Tuple

import zmq
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from LinkerHand.linker_hand_api import LinkerHandApi

# Import calculation functions from l20_controller
try:
    from l20_controller import (
        calculate_all_joint_angles,
        parse_landmarks,
    )
except ImportError:
    print("Error: Could not import from l20_controller.py")
    print("Make sure l20_controller.py is in the same directory.")
    sys.exit(1)

from debug import calculate_middle_tip_debug


class CalibrationData:
    """Stores min/max calibration values for each motor."""

    def __init__(self, config_path: Optional[str] = None):
        # Complete L20 motor list (20 motors, indices 0-19)
        self.motor_names = [
            # Base flexion (0-4)
            'thumb_base',      # 0
            'index_base',      # 1
            'middle_base',     # 2
            'ring_base',       # 3
            'pinky_base',      # 4

            # Abduction/Spread (5-9)
            'thumb_abduction', # 5
            'index_spread',    # 6
            'middle_spread',   # 7
            'ring_spread',     # 8
            'pinky_spread',    # 9

            # Thumb yaw + Reserved (10-14)
            'thumb_yaw',       # 10
            'reserved_11',     # 11
            'reserved_12',     # 12
            'reserved_13',     # 13
            'reserved_14',     # 14

            # Tip flexion (15-19)
            'thumb_tip',       # 15
            'index_tip',       # 16
            'middle_tip',      # 17
            'ring_tip',        # 18
            'pinky_tip',       # 19
        ]

        # Set config file path
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "LinkerHand",
                "config",
                "calibration_data.yaml"
            )
        self.config_path = config_path

        # Initialize with default ranges (0 to π radians for all motors)
        self.min_values: Dict[str, float] = {name: 0.0 for name in self.motor_names}
        self.max_values: Dict[str, float] = {name: math.pi for name in self.motor_names}
        # Motor output range (defaults to full 0-255 until overridden)
        self.min_motor_values: Dict[str, int] = {name: 255 for name in self.motor_names}
        self.max_motor_values: Dict[str, int] = {name: 0 for name in self.motor_names}

        # Non-linear calibration points for thumb_abduction
        # List of (motor_value, angle) tuples, sorted by motor_value
        self.thumb_abd_calibration_points: List[Tuple[int, float]] = []

    def set_motor_limits_from_pose(self, open_pose: List[int], fist_pose: List[int]):
        """Set per-motor output limits using open and fist poses."""
        if len(open_pose) != len(self.motor_names) or len(fist_pose) != len(self.motor_names):
            return
        for idx, name in enumerate(self.motor_names):
            self.min_motor_values[name] = int(open_pose[idx])
            self.max_motor_values[name] = int(fist_pose[idx])

    def set_spread_motor_limits(self, is_open: bool):
        """
        Set spread motor limits using fixed 0-255 values.

        Open: index/middle -> min (0), ring/pinky -> max (255)
        Close: index/middle -> max (255), ring/pinky -> min (0)
        """
        if is_open:
            self.min_motor_values['index_spread'] = 0
            self.min_motor_values['middle_spread'] = 0
            self.min_motor_values['ring_spread'] = 255
            self.min_motor_values['pinky_spread'] = 255
        else:
            self.max_motor_values['index_spread'] = 255
            self.max_motor_values['middle_spread'] = 255
            self.max_motor_values['ring_spread'] = 0
            self.max_motor_values['pinky_spread'] = 0

    def calibrate_min(self, current_angles: Dict[str, float]):
        """Set current angles as minimum calibration values."""
        for name in self.motor_names:
            if name in current_angles:
                self.min_values[name] = current_angles[name]

    def calibrate_max(self, current_angles: Dict[str, float]):
        """Set current angles as maximum calibration values."""
        for name in self.motor_names:
            if name in current_angles:
                self.max_values[name] = current_angles[name]

    def get_motor_value(self, motor_name: str, angle: float) -> int:
        """
        Map angle to motor value based on calibrated range and motor limits.

        For thumb_abduction with calibration points, uses non-linear interpolation.
        Otherwise uses standard linear mapping.

        Logic:
        - Min Angle -> Motor open pose value
        - Max Angle -> Motor fist pose value
        - Values are clamped to [min_motor, max_motor] range
        """
        # Use non-linear calibration for thumb_abduction if points exist
        if motor_name == 'thumb_abduction' and len(self.thumb_abd_calibration_points) >= 2:
            return self._get_motor_value_nonlinear(angle)

        # Standard linear calibration
        min_val = self.min_values.get(motor_name, 0.0)
        max_val = self.max_values.get(motor_name, math.pi)
        min_motor = self.min_motor_values.get(motor_name, 255)
        max_motor = self.max_motor_values.get(motor_name, 0)

        if abs(max_val - min_val) < 0.0001:
            # Avoid division by zero, return open pose value
            return int(min_motor)

        # Normalize to 0.0 (at min) to 1.0 (at max)
        normalized = (angle - min_val) / (max_val - min_val)

        # Clamp to 0.0 - 1.0
        clamped = max(0.0, min(1.0, normalized))

        # Map: 0.0 (Min Angle) -> open pose, 1.0 (Max Angle) -> fist pose
        return int(round(min_motor + (max_motor - min_motor) * clamped))

    def _get_motor_value_nonlinear(self, angle: float) -> int:
        """
        Get motor value using non-linear interpolation from calibration points.

        Maps angles to motor values based on calibration points.
        Points are sorted by motor value (ascending) and interpolated accordingly.
        """
        # Sort points by motor value (ascending)
        sorted_points = sorted(self.thumb_abd_calibration_points, key=lambda p: p[0])

        # Check for exact match first
        for motor, cal_angle in sorted_points:
            if abs(angle - cal_angle) < 0.0001:
                return motor

        # Check if angle is outside the calibrated range
        angles = [cal_angle for _, cal_angle in sorted_points]
        min_angle = min(angles)
        max_angle = max(angles)

        if angle <= min_angle:
            # Return motor value for minimum angle
            for motor, cal_angle in sorted_points:
                if cal_angle == min_angle:
                    return motor

        if angle >= max_angle:
            # Return motor value for maximum angle
            for motor, cal_angle in sorted_points:
                if cal_angle == max_angle:
                    return motor

        # Find two consecutive motor points where angle falls between their angles
        for i in range(len(sorted_points) - 1):
            motor1, angle1 = sorted_points[i]
            motor2, angle2 = sorted_points[i + 1]

            # Check if current angle falls between these two calibration angles
            if (angle1 >= angle >= angle2) or (angle1 <= angle <= angle2):
                # Avoid division by zero
                if abs(angle2 - angle1) < 0.0001:
                    return motor1

                # Linear interpolation
                t = (angle - angle1) / (angle2 - angle1)
                motor_value = motor1 + t * (motor2 - motor1)
                return int(round(motor_value))

        # Fallback: return nearest motor value
        return sorted_points[0][0]

    def save_to_file(self) -> bool:
        """
        Save calibration data to YAML file.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Exclude thumb_abduction from linear calibration (uses stepped calibration instead)
            min_vals = {k: v for k, v in self.min_values.items() if k != 'thumb_abduction'}
            max_vals = {k: v for k, v in self.max_values.items() if k != 'thumb_abduction'}
            min_motor_vals = {k: v for k, v in self.min_motor_values.items() if k != 'thumb_abduction'}
            max_motor_vals = {k: v for k, v in self.max_motor_values.items() if k != 'thumb_abduction'}

            calibration_dict = {
                'min_values': min_vals,
                'max_values': max_vals,
                'min_motor_values': min_motor_vals,
                'max_motor_values': max_motor_vals,
                'thumb_abd_calibration_points': [
                    {'motor': int(motor), 'angle': float(angle)}
                    for motor, angle in self.thumb_abd_calibration_points
                ],
            }

            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.safe_dump(calibration_dict, file, allow_unicode=True, default_flow_style=False)

            return True
        except Exception as e:
            print(f"Error saving calibration data: {e}")
            return False

    def load_from_file(self) -> bool:
        """
        Load calibration data from YAML file.

        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(self.config_path):
            print(f"Calibration file not found: {self.config_path}")
            return False

        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                calibration_dict = yaml.safe_load(file)

            if calibration_dict is None:
                print("Calibration file is empty")
                return False

            # Load values, validate keys exist
            if 'min_values' in calibration_dict:
                self.min_values = calibration_dict['min_values']
            if 'max_values' in calibration_dict:
                self.max_values = calibration_dict['max_values']
            if 'min_motor_values' in calibration_dict:
                self.min_motor_values = calibration_dict['min_motor_values']
            if 'max_motor_values' in calibration_dict:
                self.max_motor_values = calibration_dict['max_motor_values']
            if 'thumb_abd_calibration_points' in calibration_dict:
                points_data = calibration_dict['thumb_abd_calibration_points']
                self.thumb_abd_calibration_points = [
                    (int(p['motor']), float(p['angle']))
                    for p in points_data
                ]
                # Sort by motor value
                self.thumb_abd_calibration_points.sort(key=lambda x: x[0])

            return True
        except Exception as e:
            print(f"Error loading calibration data: {e}")
            return False


class L20ControllerGUI:
    """GUI for L20 hand controller with calibration."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("L20 Hand Controller - Calibration Interface")
        self.root.geometry("1800x1200")

        self.calibration = CalibrationData()

        # Try to load saved calibration
        if self.calibration.load_from_file():
            print(f"Loaded calibration from {self.calibration.config_path}")
        else:
            print("Using default calibration values")

        self.current_angles: Dict[str, float] = {}
        self.current_raw_angles: Dict[str, float] = {}
        self.current_pose: List[int] = [255] * 20
        self.smoothed_pose: List[int] = [255] * 20
        self.smoothing_alpha = 0.3

        # Track which motors are enabled for sending to robot
        self.motor_enabled: Dict[str, bool] = {
            name: False for name in self.calibration.motor_names
        }

        # Track last sent pose to maintain positions of disabled joints
        self.last_sent_pose: List[int] = [255] * 20

        # Preset gesture poses
        self.preset_poses = {
            "握拳": [40, 0, 0, 0, 0, 131, 10, 100, 180, 240, 19, 255, 255, 255, 255, 135, 0, 0, 0, 0],
            "张开": [255, 255, 255, 255, 255, 255, 10, 100, 180, 240, 245, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            "OK": [191, 95, 255, 255, 255, 136, 107, 100, 180, 240, 72, 255, 255, 255, 255, 116, 99, 255, 255, 255],
            "点赞": [255, 0, 0, 0, 0, 127, 10, 100, 180, 240, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0],
        }
        self._apply_preset_motor_limits()

        self.motor_angle_map = self._load_motor_angle_map()

        self.debug_data = {
            "middle_mcp": (0.0, 0.0, 0.0),
            "middle_dip": (0.0, 0.0, 0.0),
            "middle_tip": (0.0, 0.0, 0.0),
            "index_mcp": (0.0, 0.0, 0.0),
            "pinky_mcp": (0.0, 0.0, 0.0),
            "vec1": (0.0, 0.0, 0.0),
            "vec2": (0.0, 0.0, 0.0),
        }
        self.debug_vars = {
            "middle_mcp": tk.StringVar(value="(0.0000, 0.0000, 0.0000)"),
            "middle_dip": tk.StringVar(value="(0.0000, 0.0000, 0.0000)"),
            "middle_tip": tk.StringVar(value="(0.0000, 0.0000, 0.0000)"),
            "index_mcp": tk.StringVar(value="(0.0000, 0.0000, 0.0000)"),
            "pinky_mcp": tk.StringVar(value="(0.0000, 0.0000, 0.0000)"),
            "vec1": tk.StringVar(value="(0.0000, 0.0000, 0.0000)"),
            "vec2": tk.StringVar(value="(0.0000, 0.0000, 0.0000)"),
        }

        # Debug canvas for visualization
        self.debug_canvas = None

        # Hand instance reference for preset commands
        self._hand_instance = None

        self.running = False
        self.control_thread: Optional[threading.Thread] = None

        self._setup_ui()

    def _setup_ui(self):
        """Create the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text="L20 Hand Controller",
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 10))

        # Control buttons
        self._create_control_panel(main_frame)

        # Tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        angles_tab = ttk.Frame(self.notebook)
        debug_tab = ttk.Frame(self.notebook)
        thumb_abd_cal_tab = ttk.Frame(self.notebook)

        self.notebook.add(angles_tab, text="Joint Angles")
        self.notebook.add(debug_tab, text="Debug")
        self.notebook.add(thumb_abd_cal_tab, text="Thumb Abd Calibration")

        angles_tab.columnconfigure(0, weight=1)
        angles_tab.rowconfigure(0, weight=1)
        debug_tab.columnconfigure(0, weight=1)
        debug_tab.rowconfigure(0, weight=1)
        thumb_abd_cal_tab.columnconfigure(0, weight=1)
        thumb_abd_cal_tab.rowconfigure(0, weight=1)

        # Joint angles display
        self._create_angles_display(angles_tab)
        self._create_debug_display(debug_tab)
        self._create_thumb_abd_calibration_display(thumb_abd_cal_tab)

        # Status bar
        self.status_label = ttk.Label(main_frame, text="Status: Ready",
                                      relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

    def _create_control_panel(self, parent: ttk.Frame):
        """Create control buttons panel."""
        control_frame = ttk.LabelFrame(parent, text="Control", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Start/Stop button
        self.start_button = ttk.Button(control_frame, text="Start Controller",
                                       command=self.start_controller, width=20)
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = ttk.Button(control_frame, text="Stop Controller",
                                      command=self.stop_controller, width=20,
                                      state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)

        # Calibration buttons
        ttk.Separator(control_frame, orient=tk.VERTICAL).grid(
            row=0, column=2, sticky=(tk.N, tk.S), padx=10
        )

        self.cal_min_button = ttk.Button(control_frame, text="Calibrate Open",
                                         command=self.calibrate_min, width=20,
                                         state=tk.DISABLED)
        self.cal_min_button.grid(row=0, column=3, padx=5)

        self.cal_max_button = ttk.Button(control_frame, text="Calibrate Grip",
                                         command=self.calibrate_max, width=20,
                                         state=tk.DISABLED)
        self.cal_max_button.grid(row=0, column=4, padx=5)

        self.cal_spread_open_button = ttk.Button(
            control_frame, text="Calibrate Spread Open",
            command=self.calibrate_spread_min, width=22,
            state=tk.DISABLED
        )
        self.cal_spread_open_button.grid(row=0, column=5, padx=5)

        self.cal_spread_close_button = ttk.Button(
            control_frame, text="Calibrate Spread Close",
            command=self.calibrate_spread_max, width=22,
            state=tk.DISABLED
        )
        self.cal_spread_close_button.grid(row=0, column=6, padx=5)

        # Save/Reset calibration buttons
        self.save_cal_button = ttk.Button(control_frame, text="Save Calibration",
                                          command=self.save_calibration, width=20)
        self.save_cal_button.grid(row=0, column=7, padx=5)

        self.reset_cal_button = ttk.Button(control_frame, text="Reset Calibration",
                                           command=self.reset_calibration, width=20)
        self.reset_cal_button.grid(row=0, column=8, padx=5)

    def _create_angles_display(self, parent: ttk.Frame):
        """Create joint angles display table."""
        display_frame = ttk.LabelFrame(parent, text="Joint Angles", padding="10")
        display_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)

        # Create style for taller rows
        style = ttk.Style()
        style.configure("Treeview", rowheight=28, font=('Arial', 10))
        style.configure("Treeview.Heading", font=('Arial', 10, 'bold'))

        # Create Treeview
        columns = ('enabled', 'recv_deg', 'raw_deg', 'cal_min', 'cal_max', 'remapped_deg', 'motor', 'goto_min', 'goto_max')
        self.tree = ttk.Treeview(display_frame, columns=columns, height=22)

        # Define headings
        self.tree.heading('#0', text='Joint')
        self.tree.heading('enabled', text='Enabled')
        self.tree.heading('recv_deg', text='Recv (°)')
        self.tree.heading('raw_deg', text='Sent (val)')
        self.tree.heading('cal_min', text='Min (°)')
        self.tree.heading('cal_max', text='Max (°)')
        self.tree.heading('remapped_deg', text='Map (°)')
        self.tree.heading('motor', text='Motor')
        self.tree.heading('goto_min', text='Go Min')
        self.tree.heading('goto_max', text='Go Max')

        # Define column widths - increased for better visibility
        self.tree.column('#0', width=140, anchor=tk.W)
        self.tree.column('enabled', width=60, anchor=tk.CENTER)
        self.tree.column('recv_deg', width=80, anchor=tk.CENTER)
        self.tree.column('raw_deg', width=80, anchor=tk.CENTER)
        self.tree.column('cal_min', width=75, anchor=tk.CENTER)
        self.tree.column('cal_max', width=75, anchor=tk.CENTER)
        self.tree.column('remapped_deg', width=75, anchor=tk.CENTER)
        self.tree.column('motor', width=75, anchor=tk.CENTER)
        self.tree.column('goto_min', width=75, anchor=tk.CENTER)
        self.tree.column('goto_max', width=75, anchor=tk.CENTER)

        # Scrollbar
        scrollbar = ttk.Scrollbar(display_frame, orient=tk.VERTICAL,
                                  command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Add Activate/Deactivate All buttons
        button_frame = ttk.Frame(display_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0), sticky=tk.W)

        self.activate_all_button = ttk.Button(button_frame, text="Activate All Joints",
                                              command=self.activate_all_joints, width=20)
        self.activate_all_button.grid(row=0, column=0, padx=5)

        self.deactivate_all_button = ttk.Button(button_frame, text="Deactivate All Joints",
                                                command=self.deactivate_all_joints, width=20)
        self.deactivate_all_button.grid(row=0, column=1, padx=5)

        self.toggle_fingers_button = ttk.Button(
            button_frame,
            text="Toggle Finger Joints",
            command=self.toggle_finger_joints,
            width=20
        )
        self.toggle_fingers_button.grid(row=0, column=2, padx=5)

        self.disable_spread_button = ttk.Button(
            button_frame,
            text="Disable Spread Joints",
            command=self.toggle_spread_joints,
            width=20
        )
        self.disable_spread_button.grid(row=0, column=3, padx=5)

        # Add Preset Gestures label and buttons
        preset_label = ttk.Label(button_frame, text="Preset Gestures:", font=('Arial', 10, 'bold'))
        preset_label.grid(row=1, column=0, columnspan=2, pady=(10, 5), sticky=tk.W)

        self.preset_fist_button = ttk.Button(button_frame, text="握拳 (Fist)",
                                             command=lambda: self.apply_preset("握拳"), width=15)
        self.preset_fist_button.grid(row=2, column=0, padx=5, pady=2)

        self.preset_open_button = ttk.Button(button_frame, text="张开 (Open)",
                                             command=lambda: self.apply_preset("张开"), width=15)
        self.preset_open_button.grid(row=2, column=1, padx=5, pady=2)

        self.preset_ok_button = ttk.Button(button_frame, text="OK",
                                           command=lambda: self.apply_preset("OK"), width=15)
        self.preset_ok_button.grid(row=3, column=0, padx=5, pady=2)

        self.preset_thumbsup_button = ttk.Button(button_frame, text="点赞 (Thumbs Up)",
                                                 command=lambda: self.apply_preset("点赞"), width=15)
        self.preset_thumbsup_button.grid(row=3, column=1, padx=5, pady=2)

        # Populate joint names
        self._populate_joint_tree()

    def _create_debug_display(self, parent: ttk.Frame):
        """Create debug display tab."""
        # Configure parent grid for 2-column layout
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        # Left side: Text data
        debug_frame = ttk.LabelFrame(parent, text="Middle Finger Tip Angle Debug", padding="10")
        debug_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        debug_frame.columnconfigure(1, weight=1)

        labels = [
            ("Middle MCP:", "middle_mcp"),
            ("Middle DIP:", "middle_dip"),
            ("Middle TIP:", "middle_tip"),
            ("Index MCP:", "index_mcp"),
            ("Pinky MCP:", "pinky_mcp"),
            ("Vec1 (DIP→TIP):", "vec1"),
            ("Vec2 (MCP→DIP):", "vec2"),
        ]

        for row, (label, key) in enumerate(labels):
            ttk.Label(debug_frame, text=label).grid(
                row=row, column=0, sticky=tk.W, padx=(0, 10), pady=(0, 6)
            )
            ttk.Label(debug_frame, textvariable=self.debug_vars[key], font=('Arial', 10, 'bold')).grid(
                row=row, column=1, sticky=tk.W, pady=(0, 6)
            )

        # Right side: Visual canvas
        canvas_frame = ttk.LabelFrame(parent, text="Vector Visualization", padding="10")
        canvas_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))

        self.debug_canvas = tk.Canvas(canvas_frame, width=400, height=400, bg='white', relief=tk.SUNKEN, borderwidth=2)
        self.debug_canvas.pack(fill=tk.BOTH, expand=True)

    def _create_thumb_abd_calibration_display(self, parent: ttk.Frame):
        """Create thumb abduction non-linear calibration tab."""
        # Configure parent grid for 2-column layout
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=2)
        parent.rowconfigure(0, weight=1)

        # Left side: Controls
        control_frame = ttk.LabelFrame(parent, text="Real-time Motor Control", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))

        # Current angle display
        ttk.Label(control_frame, text="Current Thumb Abd Angle:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky=tk.W, pady=(0, 5)
        )
        self.thumb_abd_current_angle_var = tk.StringVar(value="0.00°")
        ttk.Label(control_frame, textvariable=self.thumb_abd_current_angle_var, font=('Arial', 12)).grid(
            row=1, column=0, sticky=tk.W, pady=(0, 15)
        )

        # Motor value slider with live preview
        ttk.Label(control_frame, text="Motor Value Slider (0-255):", font=('Arial', 10, 'bold')).grid(
            row=2, column=0, sticky=tk.W, pady=(15, 5)
        )

        # Current motor value display
        self.thumb_abd_motor_value_var = tk.StringVar(value="Motor: 0")
        ttk.Label(control_frame, textvariable=self.thumb_abd_motor_value_var, font=('Arial', 11)).grid(
            row=3, column=0, sticky=tk.W, pady=(0, 5)
        )

        # Slider for motor control
        self.thumb_abd_motor_slider = tk.Scale(
            control_frame, from_=0, to=255, orient=tk.HORIZONTAL,
            length=300, command=self.on_thumb_abd_slider_change
        )
        self.thumb_abd_motor_slider.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        self.thumb_abd_motor_slider.set(128)  # Start at middle

        # Instructions
        ttk.Label(control_frame, text="Drag slider to move thumb in real-time,\nthen capture position:",
                  font=('Arial', 9), foreground='gray').grid(
            row=5, column=0, sticky=tk.W, pady=(0, 10)
        )

        # Capture calibration point button (large, prominent)
        self.capture_cal_point_button = ttk.Button(
            control_frame, text="📍 Capture This Point",
            command=self.capture_thumb_abd_calibration_point, width=25
        )
        self.capture_cal_point_button.grid(row=6, column=0, pady=10)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(
            row=7, column=0, sticky=(tk.W, tk.E), pady=15
        )

        # Delete selected point button
        self.delete_cal_point_button = ttk.Button(
            control_frame, text="Delete Selected Point",
            command=self.delete_thumb_abd_calibration_point, width=25
        )
        self.delete_cal_point_button.grid(row=8, column=0, pady=5)

        # Clear all points button
        self.clear_cal_points_button = ttk.Button(
            control_frame, text="Clear All Points",
            command=self.clear_thumb_abd_calibration_points, width=25
        )
        self.clear_cal_points_button.grid(row=9, column=0, pady=5)

        # Right side: Calibration points table
        right_frame = ttk.Frame(parent)
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)

        # Calibration points table
        table_frame = ttk.LabelFrame(right_frame, text="Calibration Points", padding="10")
        table_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        # Create Treeview for calibration points
        columns = ('motor', 'angle')
        self.thumb_abd_tree = ttk.Treeview(table_frame, columns=columns, height=20, show='headings')

        self.thumb_abd_tree.heading('motor', text='Motor Value')
        self.thumb_abd_tree.heading('angle', text='Angle (°)')

        self.thumb_abd_tree.column('motor', width=120, anchor=tk.CENTER)
        self.thumb_abd_tree.column('angle', width=120, anchor=tk.CENTER)

        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.thumb_abd_tree.yview)
        self.thumb_abd_tree.configure(yscrollcommand=scrollbar.set)

        self.thumb_abd_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Populate table with existing points
        self.update_thumb_abd_calibration_display()

    def _format_vector(self, v: Tuple[float, float, float]) -> str:
        return f"({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})"

    def _project_to_2d(self, point_3d: Tuple[float, float, float],
                       scale: float = 1.0) -> Tuple[float, float]:
        """
        Project 3D hand landmark to 2D coordinates (not yet centered on canvas).

        Uses top-down view (X-Z plane, Y is depth).
        """
        x, y, z = point_3d

        # Project to 2D (X-Z plane, flip Z for natural orientation)
        canvas_x = x * scale
        canvas_y = -z * scale  # Flip for natural orientation

        return (canvas_x, canvas_y)

    def _load_motor_angle_map(self) -> Dict[str, str]:
        """Load motor-to-raw-joint mapping from YAML config."""
        default_map = {
            "thumb_base": "thumb_mcp",
            "thumb_abduction": "thumb_cmc_abd",
            "thumb_yaw": "thumb_cmc_flex",
            "thumb_tip": "thumb_ip",
            "index_base": "index_mcp",
            "index_tip": "index_dip",
            "middle_base": "middle_mcp",
            "middle_tip": "middle_dip",
            "ring_base": "ring_mcp",
            "ring_tip": "ring_dip",
            "pinky_base": "pinky_mcp",
            "pinky_tip": "pinky_dip",
        }

        map_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "LinkerHand",
            "config",
            "motor_angle_map.yaml",
        )
        try:
            with open(map_path, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file) or {}
            loaded_map = data.get("motor_angle_map", {})
            if isinstance(loaded_map, dict) and loaded_map:
                return loaded_map
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Failed to load motor angle map: {exc}")

        return default_map

    def _draw_debug_visualization(self):
        """Draw hand landmarks and vectors on debug canvas."""
        if not self.debug_data or not hasattr(self, 'debug_canvas') or self.debug_canvas is None:
            return

        # Clear canvas
        self.debug_canvas.delete("all")

        canvas_width = self.debug_canvas.winfo_width() or 400
        canvas_height = self.debug_canvas.winfo_height() or 400

        # Get data
        middle_mcp = self.debug_data.get("middle_mcp", (0, 0, 0))
        middle_dip = self.debug_data.get("middle_dip", (0, 0, 0))
        middle_tip = self.debug_data.get("middle_tip", (0, 0, 0))
        index_mcp = self.debug_data.get("index_mcp", (0, 0, 0))
        pinky_mcp = self.debug_data.get("pinky_mcp", (0, 0, 0))

        # Scale factor
        scale = canvas_width * 4.0

        # Project all points to 2D
        mcp_2d = self._project_to_2d(middle_mcp, scale)
        dip_2d = self._project_to_2d(middle_dip, scale)
        tip_2d = self._project_to_2d(middle_tip, scale)
        index_2d = self._project_to_2d(index_mcp, scale)
        pinky_2d = self._project_to_2d(pinky_mcp, scale)

        # Calculate rotation to make index-pinky horizontal
        # Vector from pinky to index (should be horizontal)
        dx = index_2d[0] - pinky_2d[0]
        dy = index_2d[1] - pinky_2d[1]
        angle = math.atan2(dy, dx)  # Current angle
        cos_a = math.cos(-angle)  # Rotate to make horizontal
        sin_a = math.sin(-angle)

        # Rotate all points around middle MCP
        def rotate_point(p, center):
            # Translate to origin
            x = p[0] - center[0]
            y = p[1] - center[1]
            # Rotate
            x_rot = x * cos_a - y * sin_a
            y_rot = x * sin_a + y * cos_a
            return (x_rot, y_rot)

        mcp_rot = rotate_point(mcp_2d, mcp_2d)  # (0, 0) since rotating around itself
        dip_rot = rotate_point(dip_2d, mcp_2d)
        tip_rot = rotate_point(tip_2d, mcp_2d)
        index_rot = rotate_point(index_2d, mcp_2d)
        pinky_rot = rotate_point(pinky_2d, mcp_2d)

        # Translate to canvas center
        center_x = canvas_width / 2
        center_y = canvas_height / 2

        def to_canvas(p):
            return (center_x + p[0], center_y + p[1])

        mcp_final = to_canvas(mcp_rot)
        dip_final = to_canvas(dip_rot)
        tip_final = to_canvas(tip_rot)
        index_final = to_canvas(index_rot)
        pinky_final = to_canvas(pinky_rot)

        # Draw reference points (index and pinky MCP for context)
        self.debug_canvas.create_oval(index_final[0]-4, index_final[1]-4,
                                       index_final[0]+4, index_final[1]+4,
                                       fill='lightgray', outline='gray')

        self.debug_canvas.create_oval(pinky_final[0]-4, pinky_final[1]-4,
                                       pinky_final[0]+4, pinky_final[1]+4,
                                       fill='lightgray', outline='gray')

        # Draw middle finger joints
        self.debug_canvas.create_oval(mcp_final[0]-6, mcp_final[1]-6,
                                       mcp_final[0]+6, mcp_final[1]+6,
                                       fill='blue', outline='darkblue', width=2)

        self.debug_canvas.create_oval(dip_final[0]-6, dip_final[1]-6,
                                       dip_final[0]+6, dip_final[1]+6,
                                       fill='green', outline='darkgreen', width=2)

        self.debug_canvas.create_oval(tip_final[0]-6, tip_final[1]-6,
                                       tip_final[0]+6, tip_final[1]+6,
                                       fill='red', outline='darkred', width=2)

        # Draw Vec2: MCP → DIP (purple arrow)
        self.debug_canvas.create_line(mcp_final[0], mcp_final[1],
                                       dip_final[0], dip_final[1],
                                       arrow=tk.LAST, fill='purple', width=3,
                                       arrowshape=(12, 15, 5))

        # Draw Vec1: DIP → TIP (orange arrow)
        self.debug_canvas.create_line(dip_final[0], dip_final[1],
                                       tip_final[0], tip_final[1],
                                       arrow=tk.LAST, fill='orange', width=3,
                                       arrowshape=(12, 15, 5))

    def update_debug_display(self):
        """Update debug tab values."""
        if not self.debug_data:
            return

        # Update text labels
        vector_keys = ["middle_mcp", "middle_dip", "middle_tip", "index_mcp", "pinky_mcp", "vec1", "vec2"]
        for key in vector_keys:
            self.debug_vars[key].set(
                self._format_vector(self.debug_data.get(key, (0.0, 0.0, 0.0)))
            )

        # Update canvas visualization
        self._draw_debug_visualization()

    def _sanitize_spread_angles(self, motor_angles: Dict[str, float]) -> Dict[str, float]:
        """Apply safety ordering constraints to spread angles."""
        index_val = motor_angles.get("index_spread", 0.0)
        middle_val = motor_angles.get("middle_spread", 0.0)
        ring_val = motor_angles.get("ring_spread", 0.0)
        pinky_val = motor_angles.get("pinky_spread", 0.0)

        if middle_val > ring_val:
            avg = (middle_val + ring_val) / 2.0
            middle_val = avg
            ring_val = avg

        if index_val > middle_val:
            index_val = middle_val

        if pinky_val < ring_val:
            pinky_val = ring_val

        motor_angles["index_spread"] = index_val
        motor_angles["middle_spread"] = middle_val
        motor_angles["ring_spread"] = ring_val
        motor_angles["pinky_spread"] = pinky_val
        return motor_angles

    def _populate_joint_tree(self):
        """Populate the tree with motor names grouped by finger."""
        finger_groups = [
            "Thumb",
            "Index",
            "Middle",
            "Ring",
            "Pinky",
            "Reserved",
        ]

        for finger_name in finger_groups:
            # Insert parent (finger name)
            parent = self.tree.insert('', tk.END, text=finger_name,
                                      values=('\u2713', '', '', '', '', '', '', '[MIN]', '[MAX]'),
                                      tags=('parent',))

            # Get motors for this finger (spread on top)
            motors = self._get_finger_motors(finger_name)

            # Insert children (motor names with motor index)
            for motor_name in motors:
                motor_idx = self.calibration.motor_names.index(motor_name)
                # Display as "Motor X: Description"
                motor_desc = motor_name.replace('_', ' ').title()
                display_name = f"  Motor {motor_idx}: {motor_desc}"

                self.tree.insert(parent, tk.END, iid=motor_name, text=display_name,
                                values=('\u2713', '0.0', '0.0', '0.0', '180.0', '0.0', '255', '[MIN]', '[MAX]'))

            # Expand all sections by default
            self.tree.item(parent, open=True)

        # Tag configuration for styling
        self.tree.tag_configure('parent', font=('Arial', 11, 'bold'), foreground='#0066cc')

        # Bind click event for toggling enabled state
        self.tree.bind('<Button-1>', self._on_tree_click)

    def _get_finger_motors(self, finger_name: str) -> List[str]:
        """Get list of motor names for a given finger (spread on top)."""
        finger_map = {
            "Thumb": ['thumb_abduction', 'thumb_yaw', 'thumb_base', 'thumb_tip'],
            "Index": ['index_spread', 'index_base', 'index_tip'],
            "Middle": ['middle_spread', 'middle_base', 'middle_tip'],
            "Ring": ['ring_spread', 'ring_base', 'ring_tip'],
            "Pinky": ['pinky_spread', 'pinky_base', 'pinky_tip'],
            "Reserved": ['reserved_11', 'reserved_12', 'reserved_13', 'reserved_14'],
        }
        return finger_map.get(finger_name, [])

    def _go_to_motor_limit(self, motor_name: str, is_min: bool):
        """
        Move a specific motor to limit (0 or 255).

        Args:
            motor_name: Name of the motor to move
            is_min: True to go to motor 0 (fully flexed), False to go to motor 255 (fully extended)
        """
        if not self.running or not self._hand_instance:
            self.status_label.config(text="Status: Please start controller before moving motors")
            return

        # Set motor value to 0 (min/flexed) or 255 (max/extended)
        motor_value = 0 if is_min else 255

        # Get pose index for this motor (1:1 mapping)
        motor_idx = self._get_pose_index_for_motor(motor_name)

        # Update the motor position in last_sent_pose and current_pose
        self.last_sent_pose[motor_idx] = motor_value
        self.current_pose[motor_idx] = motor_value

        # Send the updated pose
        try:
            self._hand_instance.finger_move(pose=self.last_sent_pose)
            limit_type = "MIN (0)" if is_min else "MAX (255)"
            self.status_label.config(
                text=f"Status: Moved {motor_name} (motor {motor_idx}) to {limit_type}"
            )
            # Update display immediately
            self.update_angles_display()
        except Exception as e:  # noqa: BLE001
            print(f"Error moving motor: {e}")
            self.status_label.config(text=f"Status: Error moving motor - {e}")

    def _get_pose_index_for_motor(self, motor_name: str) -> int:
        """Get pose index for a motor (1:1 mapping)."""
        # Direct 1:1 mapping - motor name index = pose index
        return self.calibration.motor_names.index(motor_name)

    def _on_tree_click(self, event):
        """Handle click events on the tree to toggle enabled state or go to min/max."""
        # Identify which column was clicked
        column = self.tree.identify_column(event.x)

        # Identify which row was clicked
        row_id = self.tree.identify_row(event.y)
        if not row_id:
            return

        # Handle "Go to Min" column (#8)
        if column == '#8':
            if row_id in self.calibration.motor_names:
                # Individual motor - go to min
                self._go_to_motor_limit(row_id, is_min=True)
            else:
                # Parent row (finger group) - go to min for all motors
                motors = self._get_finger_motors(self.tree.item(row_id)['text'])
                for motor in motors:
                    self._go_to_motor_limit(motor, is_min=True)
            return

        # Handle "Go to Max" column (#9)
        if column == '#9':
            if row_id in self.calibration.motor_names:
                # Individual motor - go to max
                self._go_to_motor_limit(row_id, is_min=False)
            else:
                # Parent row (finger group) - go to max for all motors
                motors = self._get_finger_motors(self.tree.item(row_id)['text'])
                for motor in motors:
                    self._go_to_motor_limit(motor, is_min=False)
            return

        # Handle enabled/disabled toggle (#1)
        if column != '#1':
            return

        # Check if it's a parent (finger group) or child (individual motor)
        if row_id in self.calibration.motor_names:
            # Individual motor
            self.motor_enabled[row_id] = not self.motor_enabled[row_id]
        else:
            # Parent row (finger group)
            motors = self._get_finger_motors(self.tree.item(row_id)['text'])
            # Toggle all children - if any are disabled, enable all; otherwise disable all
            any_disabled = any(not self.motor_enabled.get(m, True) for m in motors)
            new_state = True if any_disabled else False
            for motor in motors:
                self.motor_enabled[motor] = new_state

        # Update the display
        self._update_tree_enabled_display()

    def _update_tree_enabled_display(self):
        """Refresh the enabled column display for all items."""
        # Update individual motors
        for motor_name in self.calibration.motor_names:
            enabled = self.motor_enabled.get(motor_name, True)
            symbol = '\u2713' if enabled else '\u2717'  # ✓ or ✗
            current_values = list(self.tree.item(motor_name)['values'])
            current_values[0] = symbol
            try:
                self.tree.item(motor_name, values=current_values)
            except tk.TclError:
                pass

        # Update parent rows
        finger_groups = ["Thumb", "Index", "Middle", "Ring", "Pinky", "Reserved"]
        for item in self.tree.get_children():
            finger_name = self.tree.item(item)['text']
            if finger_name in finger_groups:
                motors = self._get_finger_motors(finger_name)
                enabled_count = sum(1 for m in motors if self.motor_enabled.get(m, True))
                if enabled_count == len(motors):
                    symbol = '\u2713'  # All enabled
                elif enabled_count == 0:
                    symbol = '\u2717'  # All disabled
                else:
                    symbol = '\u2212'  # Mixed (−)

                current_values = list(self.tree.item(item)['values'])
                current_values[0] = symbol
                try:
                    self.tree.item(item, values=current_values)
                except tk.TclError:
                    pass

    def activate_all_joints(self):
        """Enable all motors except the four finger spread joints."""
        four_finger_spreads = set(self._four_finger_spread_names())
        for motor_name in self.calibration.motor_names:
            if motor_name in four_finger_spreads:
                # Always keep four finger spread joints disabled
                self.motor_enabled[motor_name] = False
            else:
                self.motor_enabled[motor_name] = True
        self._update_tree_enabled_display()

        # Update status to indicate tracking resumed
        if self.running:
            self.status_label.config(text="Status: Running - MediaPipe tracking resumed (4 finger spreads disabled)")

    def deactivate_all_joints(self):
        """Disable all motors."""
        for motor_name in self.calibration.motor_names:
            self.motor_enabled[motor_name] = False
        self._update_tree_enabled_display()

    def toggle_finger_joints(self):
        """Toggle all finger joints except spreading joints."""
        excluded_joints = {
            'thumb_base',
            'thumb_abduction',
            'thumb_yaw',
            'thumb_tip',
            'index_spread',
            'middle_spread',
            'ring_spread',
            'pinky_spread',
        }
        finger_joints = [
            name for name in self.calibration.motor_names if name not in excluded_joints
        ]
        any_disabled = any(not self.motor_enabled.get(name, True) for name in finger_joints)
        new_state = True if any_disabled else False
        for name in finger_joints:
            self.motor_enabled[name] = new_state
        self._update_tree_enabled_display()

    def toggle_spread_joints(self):
        """Toggle all spread joints."""
        spread_joints = self._spread_joint_names()
        any_disabled = any(not self.motor_enabled.get(name, True) for name in spread_joints)
        new_state = True if any_disabled else False
        for name in spread_joints:
            self.motor_enabled[name] = new_state
        self._update_tree_enabled_display()
        self.disable_spread_button.config(
            text="Disable Spread Joints" if new_state else "Enable Spread Joints"
        )

    def _spread_joint_names(self) -> List[str]:
        return [
            'thumb_abduction',
            'index_spread',
            'middle_spread',
            'ring_spread',
            'pinky_spread',
        ]

    def _four_finger_spread_names(self) -> List[str]:
        """Return only the four finger spread joints (excluding thumb)."""
        return [
            'index_spread',
            'middle_spread',
            'ring_spread',
            'pinky_spread',
        ]

    def _get_open_pose_value(self, motor_name: str) -> int:
        open_pose = self.preset_poses.get("张开")
        if not open_pose:
            return 255
        motor_idx = self._get_pose_index_for_motor(motor_name)
        return int(open_pose[motor_idx])

    def apply_preset(self, gesture_name: str):
        """Apply a preset gesture pose to the hand."""
        if gesture_name not in self.preset_poses:
            return

        preset_pose = self.preset_poses[gesture_name]

        # Send preset pose to hand if controller is running
        if self.running and self._hand_instance:
            try:
                self._hand_instance.finger_move(pose=preset_pose)
            except Exception as e:  # noqa: BLE001
                print(f"Error sending preset: {e}")
        elif not self.running:
            # Show warning if controller is not running
            self.status_label.config(text="Status: Please start controller before applying presets")
            return

        # Update last sent pose and current pose
        self.last_sent_pose = preset_pose.copy()
        self.current_pose = preset_pose.copy()

        # Deactivate all joints to pause MediaPipe tracking
        self.deactivate_all_joints()

        # Update display immediately
        self.update_angles_display()

        # Update status
        self.status_label.config(text=f"Status: Preset '{gesture_name}' applied - Click 'Activate All' to resume tracking")

    def update_angles_display(self):
        """Update the angles display with current motor values."""
        # Display motor values directly from current_pose (20 motors)
        for motor_name in self.calibration.motor_names:
            motor_idx = self._get_pose_index_for_motor(motor_name)
            motor_value = self.current_pose[motor_idx]

            cal_min = self.calibration.min_values[motor_name]
            cal_max = self.calibration.max_values[motor_name]

            # Received raw joint angle mapped to this motor
            recv_angle = self._get_raw_angle_for_motor(motor_name)
            recv_deg = math.degrees(recv_angle)

            # Mapped angle (normalized recv_angle between cal_min/cal_max mapped to 0-180)
            range_val = cal_max - cal_min
            if abs(range_val) > 0.0001:
                norm = max(0.0, min(1.0, (recv_angle - cal_min) / range_val))
                map_deg = norm * 180.0
            else:
                map_deg = 0.0

            # Get enabled status
            enabled = self.motor_enabled.get(motor_name, True)
            enabled_symbol = '\u2713' if enabled else '\u2717'  # ✓ or ✗

            values = (
                enabled_symbol,
                f"{recv_deg:.1f}",
                f"{motor_value}",
                f"{math.degrees(cal_min):.1f}",
                f"{math.degrees(cal_max):.1f}",
                f"{map_deg:.1f}",
                f"{motor_value}",
                "[MIN]",
                "[MAX]"
            )

            try:
                self.tree.item(motor_name, values=values)
            except tk.TclError:
                pass  # Motor not in tree yet

        # Update parent rows to show aggregated enabled status
        finger_groups = ["Thumb", "Index", "Middle", "Ring", "Pinky", "Reserved"]
        for item in self.tree.get_children():
            finger_name = self.tree.item(item)['text']
            if finger_name in finger_groups:
                motors = self._get_finger_motors(finger_name)
                enabled_count = sum(1 for m in motors if self.motor_enabled.get(m, True))
                if enabled_count == len(motors):
                    symbol = '\u2713'  # All enabled
                elif enabled_count == 0:
                    symbol = '\u2717'  # All disabled
                else:
                    symbol = '\u2212'  # Mixed (−)

                current_values = list(self.tree.item(item)['values'])
                if current_values:
                    current_values[0] = symbol
                    try:
                        self.tree.item(item, values=current_values)
                    except tk.TclError:
                        pass

    def _get_raw_angle_for_motor(self, motor_name: str) -> float:
        """Map motor name to a raw joint angle from calculate_all_joint_angles()."""
        raw_key = self.motor_angle_map.get(motor_name)
        if not raw_key:
            return 0.0
        return self.current_raw_angles.get(raw_key, 0.0)

    def start_controller(self):
        """Start the hand controller."""
        if self.running:
            return

        # Ensure joints are deactivated on start
        self.deactivate_all_joints()

        # Always ensure four finger spread joints are disabled
        for motor_name in self._four_finger_spread_names():
            self.motor_enabled[motor_name] = False

        # Start control thread
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

        # Update UI immediately to indicate attempt
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.cal_min_button.config(state=tk.NORMAL)
        self.cal_max_button.config(state=tk.NORMAL)
        self.cal_spread_open_button.config(state=tk.NORMAL)
        self.cal_spread_close_button.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Starting...")

    def stop_controller(self):
        """Stop the hand controller."""
        self.running = False

        if self.control_thread and self.control_thread.is_alive():
            # Wait for thread to clean up and exit
            self.control_thread.join(timeout=2.0)

        # Update UI
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.cal_min_button.config(state=tk.DISABLED)
        self.cal_max_button.config(state=tk.DISABLED)
        self.cal_spread_open_button.config(state=tk.DISABLED)
        self.cal_spread_close_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped")

    def calibrate_min(self):
        """Calibrate minimum values with current angles."""
        if self.current_angles:
            # Exclude spread joints and thumb_abduction (uses stepped calibration)
            excluded_joints = {
                'index_spread',
                'middle_spread',
                'ring_spread',
                'pinky_spread',
                'thumb_abduction',
            }
            filtered_angles = {
                name: angle for name, angle in self.current_angles.items()
                if name not in excluded_joints
            }
            self.calibration.calibrate_min(filtered_angles)
            self.calibration.save_to_file()  # Auto-save
            self.status_label.config(text="Status: MIN calibration captured and saved!")
            self.update_angles_display()
            self.root.after(2000, lambda: self.status_label.config(
                text="Status: Running - Listening on tcp://localhost:5557"
            ))

    def calibrate_max(self):
        """Calibrate maximum values with current angles."""
        if self.current_angles:
            # Exclude spread joints and thumb_abduction (uses stepped calibration)
            excluded_joints = {
                'index_spread',
                'middle_spread',
                'ring_spread',
                'pinky_spread',
                'thumb_abduction',
            }
            filtered_angles = {
                name: angle for name, angle in self.current_angles.items()
                if name not in excluded_joints
            }
            self.calibration.calibrate_max(filtered_angles)
            self.calibration.save_to_file()  # Auto-save
            self.status_label.config(text="Status: MAX calibration captured and saved!")
            self.update_angles_display()
            self.root.after(2000, lambda: self.status_label.config(
                text="Status: Running - Listening on tcp://localhost:5557"
            ))

    def calibrate_spread_min(self):
        """Calibrate spread minimum values with current angles."""
        if self.current_angles:
            spread_joints = {
                'index_spread',
                'middle_spread',
                'ring_spread',
                'pinky_spread',
            }
            filtered_angles = {
                name: angle for name, angle in self.current_angles.items()
                if name in spread_joints
            }
            self.calibration.calibrate_min(filtered_angles)
            self.calibration.set_spread_motor_limits(is_open=True)
            self.calibration.save_to_file()  # Auto-save
            self.status_label.config(text="Status: Spread MIN calibration captured and saved!")
            self.update_angles_display()
            self.root.after(2000, lambda: self.status_label.config(
                text="Status: Running - Listening on tcp://localhost:5557"
            ))

    def calibrate_spread_max(self):
        """Calibrate spread maximum values with current angles."""
        if self.current_angles:
            spread_joints = {
                'index_spread',
                'middle_spread',
                'ring_spread',
                'pinky_spread',
            }
            filtered_angles = {
                name: angle for name, angle in self.current_angles.items()
                if name in spread_joints
            }
            self.calibration.calibrate_max(filtered_angles)
            self.calibration.set_spread_motor_limits(is_open=False)
            self.calibration.save_to_file()  # Auto-save
            self.status_label.config(text="Status: Spread MAX calibration captured and saved!")
            self.update_angles_display()
            self.root.after(2000, lambda: self.status_label.config(
                text="Status: Running - Listening on tcp://localhost:5557"
            ))

    def save_calibration(self):
        """Save current calibration to file."""
        if self.calibration.save_to_file():
            self.status_label.config(text=f"Status: Calibration saved to {self.calibration.config_path}")
            self.root.after(2000, lambda: self.status_label.config(
                text="Status: Ready" if not self.running else
                     "Status: Running - Listening on tcp://localhost:5557"
            ))
        else:
            self.status_label.config(text="Status: Error saving calibration!")
            self.root.after(2000, lambda: self.status_label.config(
                text="Status: Ready" if not self.running else
                     "Status: Running - Listening on tcp://localhost:5557"
            ))

    def reset_calibration(self):
        """Reset calibration to defaults."""
        # Reset all values to defaults
        for name in self.calibration.motor_names:
            self.calibration.min_values[name] = 0.0
            self.calibration.max_values[name] = math.pi
            self.calibration.min_motor_values[name] = 255
            self.calibration.max_motor_values[name] = 0

        self._apply_preset_motor_limits()
        self.status_label.config(text="Status: Calibration reset to defaults")
        self.update_angles_display()
        self.root.after(2000, lambda: self.status_label.config(
            text="Status: Ready" if not self.running else
                 "Status: Running - Listening on tcp://localhost:5557"
        ))

    def on_thumb_abd_slider_change(self, value):
        """Called when slider is moved - sends motor command in real-time."""
        motor_value = int(float(value))
        self.thumb_abd_motor_value_var.set(f"Motor: {motor_value}")

        # Only send commands if the Thumb Abd Calibration tab is active
        current_tab = self.notebook.tab(self.notebook.select(), "text")
        if current_tab != "Thumb Abd Calibration":
            return

        # Send motor command to hand if controller is running
        if self.running and self._hand_instance:
            try:
                # Get current pose and update only thumb_abduction motor
                motor_idx = self._get_pose_index_for_motor('thumb_abduction')
                temp_pose = self.last_sent_pose.copy()
                temp_pose[motor_idx] = motor_value

                # Send to hand
                self._hand_instance.finger_move(pose=temp_pose)

                # Update last_sent_pose so other motors maintain their position
                self.last_sent_pose[motor_idx] = motor_value
            except Exception as e:
                print(f"Error sending slider motor command: {e}")

    def capture_thumb_abd_calibration_point(self):
        """Capture current slider position and angle as a calibration point."""
        # Get motor value from slider
        motor_value = int(self.thumb_abd_motor_slider.get())

        # Get current angle
        if 'thumb_abduction' not in self.current_angles:
            self.status_label.config(text="Status: No thumb abduction angle available - Start controller first")
            return

        current_angle = self.current_angles['thumb_abduction']

        # Remove existing point with same motor value
        self.calibration.thumb_abd_calibration_points = [
            (m, a) for m, a in self.calibration.thumb_abd_calibration_points
            if m != motor_value
        ]

        # Add new point (store in radians internally)
        self.calibration.thumb_abd_calibration_points.append((motor_value, current_angle))

        # Sort by motor value
        self.calibration.thumb_abd_calibration_points.sort(key=lambda x: x[0])

        # Save and update display
        self.calibration.save_to_file()
        self.update_thumb_abd_calibration_display()

        current_angle_deg = math.degrees(current_angle)
        self.status_label.config(
            text=f"Status: ✓ Captured point - Motor={motor_value}, Angle={current_angle_deg:.2f}°"
        )
        self.root.after(2000, lambda: self.status_label.config(
            text="Status: Running - Listening on tcp://localhost:5557" if self.running else "Status: Ready"
        ))

    def delete_thumb_abd_calibration_point(self):
        """Delete selected calibration point."""
        selection = self.thumb_abd_tree.selection()
        if not selection:
            self.status_label.config(text="Status: No point selected")
            return

        # Get selected motor value
        item = self.thumb_abd_tree.item(selection[0])
        motor_value = int(item['values'][0])

        # Remove point
        self.calibration.thumb_abd_calibration_points = [
            (m, a) for m, a in self.calibration.thumb_abd_calibration_points
            if m != motor_value
        ]

        # Save and update display
        self.calibration.save_to_file()
        self.update_thumb_abd_calibration_display()

        self.status_label.config(text=f"Status: Deleted point at motor={motor_value}")
        self.root.after(2000, lambda: self.status_label.config(
            text="Status: Running - Listening on tcp://localhost:5557" if self.running else "Status: Ready"
        ))

    def clear_thumb_abd_calibration_points(self):
        """Clear all calibration points."""
        self.calibration.thumb_abd_calibration_points = []
        self.calibration.save_to_file()
        self.update_thumb_abd_calibration_display()

        self.status_label.config(text="Status: All calibration points cleared")
        self.root.after(2000, lambda: self.status_label.config(
            text="Status: Running - Listening on tcp://localhost:5557" if self.running else "Status: Ready"
        ))

    def update_thumb_abd_calibration_display(self):
        """Update the thumb abduction calibration display."""
        # Clear existing items
        for item in self.thumb_abd_tree.get_children():
            self.thumb_abd_tree.delete(item)

        # Add calibration points (convert radians to degrees for display)
        for motor, angle in self.calibration.thumb_abd_calibration_points:
            angle_deg = math.degrees(angle)
            self.thumb_abd_tree.insert('', tk.END, values=(motor, f"{angle_deg:.2f}"))

        # Update current angle if available (convert to degrees)
        if 'thumb_abduction' in self.current_angles:
            current_angle = self.current_angles['thumb_abduction']
            current_angle_deg = math.degrees(current_angle)
            self.thumb_abd_current_angle_var.set(f"{current_angle_deg:.2f}°")

    def _apply_preset_motor_limits(self):
        """Apply open/fist presets as per-motor output limits."""
        open_pose = self.preset_poses.get("张开")
        fist_pose = self.preset_poses.get("握拳")
        if open_pose and fist_pose:
            self.calibration.set_motor_limits_from_pose(open_pose, fist_pose)

    def _merge_poses(self, new_pose: List[int]) -> List[int]:
        """
        Merge new pose with last sent pose based on enabled motors.

        For enabled motors, use values from new_pose.
        For disabled motors, keep values from last_sent_pose (frozen).
        Exception: Non-thumb spread joints go to open pose when disabled.

        Args:
            new_pose: The newly calculated pose (20 motor values)

        Returns:
            Merged pose with enabled motors updated, disabled motors preserved
        """
        # Start with a copy of the last sent pose
        merged_pose = self.last_sent_pose.copy()

        # Update pose index for each motor (1:1 mapping)
        for motor_name in self.calibration.motor_names:
            motor_idx = self._get_pose_index_for_motor(motor_name)
            if self.motor_enabled.get(motor_name, True):
                merged_pose[motor_idx] = new_pose[motor_idx]
            elif motor_name in self._spread_joint_names() and not motor_name.startswith('thumb'):
                # Non-thumb spread joints go to open pose when disabled
                merged_pose[motor_idx] = self._get_open_pose_value(motor_name)
            # else: keep last_sent_pose value (frozen)

        return merged_pose

    def _control_loop(self):
        """Main control loop running in separate thread."""
        zmq_socket = None
        hand = None

        try:
            # Initialize ZMQ socket
            context = zmq.Context.instance()
            zmq_socket = context.socket(zmq.SUB)
            zmq_socket.connect("tcp://localhost:5557")
            zmq_socket.setsockopt(zmq.SUBSCRIBE, b"")
            zmq_socket.RCVTIMEO = 500

            # Initialize hand
            hand = LinkerHandApi(hand_type="right", hand_joint="G20")
            hand.set_speed([30, 30, 30, 30, 30])

            # Store hand instance for preset commands
            self._hand_instance = hand

            # Update status to success
            self.root.after(0, lambda: self.status_label.config(
                text="Status: Running - Listening on tcp://localhost:5557"
            ))

            while self.running:
                # Receive landmarks
                try:
                    message = zmq_socket.recv_string()
                except zmq.Again:
                    continue
                except zmq.ZMQError:
                    continue

                # Parse and calculate angles
                try:
                    landmarks = parse_landmarks(message)
                    raw_angles = calculate_all_joint_angles(landmarks)
                    self.current_raw_angles = raw_angles

                    try:
                        self.debug_data = calculate_middle_tip_debug(landmarks)
                        self.root.after(0, self.update_debug_display)
                    except Exception:
                        pass

                    # Map raw joint angles directly to motor angles
                    motor_angles = {}
                    for motor_name in self.calibration.motor_names:
                        motor_angles[motor_name] = self._get_raw_angle_for_motor(motor_name)
                    motor_angles = self._sanitize_spread_angles(motor_angles)

                    # Store for display and calibration
                    self.current_angles = motor_angles

                    # Calculate new pose using calibrated values
                    new_pose = [0] * 20
                    for motor_name in self.calibration.motor_names:
                        angle = motor_angles.get(motor_name, 0.0)
                        motor_val = self.calibration.get_motor_value(motor_name, angle)
                        motor_idx = self._get_pose_index_for_motor(motor_name)
                        new_pose[motor_idx] = motor_val

                    # Apply EMA smoothing to enabled motors
                    smoothed_pose = self.smoothed_pose.copy()
                    alpha = self.smoothing_alpha
                    for motor_name in self.calibration.motor_names:
                        motor_idx = self._get_pose_index_for_motor(motor_name)
                        if self.motor_enabled.get(motor_name, True):
                            target = new_pose[motor_idx]
                            prev = smoothed_pose[motor_idx]
                            smoothed_pose[motor_idx] = int(round(alpha * target + (1.0 - alpha) * prev))
                        else:
                            smoothed_pose[motor_idx] = self.last_sent_pose[motor_idx]
                    self.smoothed_pose = smoothed_pose

                    # Ensure four finger spread joints are always disabled
                    for motor_name in self._four_finger_spread_names():
                        self.motor_enabled[motor_name] = False

                    # Merge with last sent pose based on enabled joints
                    self.current_pose = self._merge_poses(smoothed_pose)

                    # Update last sent pose
                    self.last_sent_pose = self.current_pose.copy()

                    # Update display (in main thread)
                    self.root.after(0, self.update_angles_display)
                    self.root.after(0, self.update_thumb_abd_calibration_display)

                    # Update status if it was previously showing "No hand"
                    if "No hand" in self.status_label.cget("text"):
                         self.root.after(0, lambda: self.status_label.config(
                            text="Status: Running - Listening on tcp://localhost:5557"
                        ))

                except ValueError as e:
                    if "No hand landmarks detected" in str(e):
                        # Update status to indicate no hand seen (throttle UI updates if needed, but simple string check is fast)
                        self.root.after(0, lambda: self.status_label.config(
                            text="Status: Connected - No hand detected"
                        ))
                        continue
                    else:
                        print(f"Parse error: {e}")
                        continue
                except Exception as e:  # noqa: BLE001
                    print(f"Parse error: {e}")
                    continue

                # Send to hand
                try:
                    if hand:
                        hand.finger_move(pose=self.current_pose)
                except Exception as e:  # noqa: BLE001
                    print(f"Hand control error: {e}")

                time.sleep(0.02)  # ~50Hz

        except Exception as e:
            print(f"Control loop init error: {e}")
            self.root.after(0, lambda err=str(e): self.status_label.config(
                text=f"Status: Error - {err}"
            ))
            self.running = False
            # Reset UI state if init failed
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))

        finally:
            # Clear hand instance reference
            self._hand_instance = None

            # Cleanup resources
            if hand:
                try:
                    hand.close_can()
                except Exception:
                    pass

            if zmq_socket:
                try:
                    zmq_socket.close()
                except Exception:
                    pass

    def on_closing(self):
        """Handle window close event."""
        self.stop_controller()
        self.root.destroy()


def main():
    """Run the GUI application."""
    root = tk.Tk()
    app = L20ControllerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
