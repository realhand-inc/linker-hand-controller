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

from debug import calculate_middle_mcp_flexion_debug


class CalibrationData:
    """Stores min/max calibration values for each motor."""

    def __init__(self):
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

        # Initialize with default ranges (0 to π radians for all motors)
        self.min_values: Dict[str, float] = {name: 0.0 for name in self.motor_names}
        self.max_values: Dict[str, float] = {name: math.pi for name in self.motor_names}

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
        Map angle to motor value (0-255) based on calibrated range.

        Logic:
        - Min Angle -> Motor 255 (Open/Extended)
        - Max Angle -> Motor 0 (Closed/Flexed)
        - Values are clamped to 0-255
        """
        min_val = self.min_values.get(motor_name, 0.0)
        max_val = self.max_values.get(motor_name, math.pi)

        if abs(max_val - min_val) < 0.0001:
            # Avoid division by zero, return default open (255)
            return 255

        # Normalize to 0.0 (at min) to 1.0 (at max)
        normalized = (angle - min_val) / (max_val - min_val)

        # Clamp to 0.0 - 1.0
        clamped = max(0.0, min(1.0, normalized))

        # Invert: 0.0 (Min Angle) -> 255, 1.0 (Max Angle) -> 0
        return int((1.0 - clamped) * 255)


class L20ControllerGUI:
    """GUI for L20 hand controller with calibration."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("L20 Hand Controller - Calibration Interface")
        self.root.geometry("1800x1200")

        self.calibration = CalibrationData()
        self.current_angles: Dict[str, float] = {}
        self.current_pose: List[int] = [255] * 20

        # Track which motors are enabled for sending to robot
        self.motor_enabled: Dict[str, bool] = {
            name: True for name in self.calibration.motor_names
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

        self.debug_data = {
            "wrist": (0.0, 0.0, 0.0),
            "index_mcp": (0.0, 0.0, 0.0),
            "pinky_mcp": (0.0, 0.0, 0.0),
            "middle_mcp": (0.0, 0.0, 0.0),
            "middle_pip": (0.0, 0.0, 0.0),
            "plane_normal": (0.0, 0.0, 0.0),
            "vec1": (0.0, 0.0, 0.0),
            "vec2": (0.0, 0.0, 0.0),
            "vec1_proj": (0.0, 0.0, 0.0),
            "vec2_proj": (0.0, 0.0, 0.0),
            "vec1_norm": (0.0, 0.0, 0.0),
            "vec2_norm": (0.0, 0.0, 0.0),
            "angle_rad": 0.0,
        }
        self.debug_vars = {
            "wrist": tk.StringVar(value="(0.0000, 0.0000, 0.0000)"),
            "index_mcp": tk.StringVar(value="(0.0000, 0.0000, 0.0000)"),
            "pinky_mcp": tk.StringVar(value="(0.0000, 0.0000, 0.0000)"),
            "middle_mcp": tk.StringVar(value="(0.0000, 0.0000, 0.0000)"),
            "middle_pip": tk.StringVar(value="(0.0000, 0.0000, 0.0000)"),
            "plane_normal": tk.StringVar(value="(0.0000, 0.0000, 0.0000)"),
            "vec1": tk.StringVar(value="(0.0000, 0.0000, 0.0000)"),
            "vec2": tk.StringVar(value="(0.0000, 0.0000, 0.0000)"),
            "vec1_proj": tk.StringVar(value="(0.0000, 0.0000, 0.0000)"),
            "vec2_proj": tk.StringVar(value="(0.0000, 0.0000, 0.0000)"),
            "vec1_norm": tk.StringVar(value="(0.0000, 0.0000, 0.0000)"),
            "vec2_norm": tk.StringVar(value="(0.0000, 0.0000, 0.0000)"),
            "angle_rad": tk.StringVar(value="0.0000 rad (0.00°)"),
        }

        # Hand instance reference for preset commands
        self._hand_instance = None

        self.running = False
        self.control_thread: Optional[threading.Thread] = None

        self._setup_ui()

    def _calculate_motor_angles(self, raw_angles: Dict[str, float]) -> Dict[str, float]:
        """
        Aggregate raw joint angles into motor-specific control angles.
        """
        motor_angles = {}

        # Helper for getting raw angle with default 0
        def get(name): return raw_angles.get(name, 0.0)

        # Thumb
        motor_angles['thumb_base'] = (get('thumb_mcp') + get('thumb_ip')) / 2.0
        motor_angles['thumb_abduction'] = get('thumb_cmc_abd')
        motor_angles['thumb_yaw'] = get('thumb_cmc_flex')
        motor_angles['thumb_tip'] = get('thumb_ip')

        # Index
        motor_angles['index_base'] = (get('index_mcp') + get('index_pip') + get('index_dip')) / 3.0
        motor_angles['index_tip'] = get('index_dip')
        motor_angles['index_spread'] = 0.0  # Default

        # Middle
        motor_angles['middle_base'] = (get('middle_mcp') + get('middle_pip') + get('middle_dip')) / 3.0
        motor_angles['middle_tip'] = get('middle_dip')
        motor_angles['middle_spread'] = 0.0

        # Ring
        motor_angles['ring_base'] = (get('ring_mcp') + get('ring_pip') + get('ring_dip')) / 3.0
        motor_angles['ring_tip'] = get('ring_dip')
        motor_angles['ring_spread'] = 0.0

        # Pinky
        motor_angles['pinky_base'] = (get('pinky_mcp') + get('pinky_pip') + get('pinky_dip')) / 3.0
        motor_angles['pinky_tip'] = get('pinky_dip')
        motor_angles['pinky_spread'] = 0.0

        # Reserved
        for i in range(11, 15):
            motor_angles[f'reserved_{i}'] = 0.0

        return motor_angles

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

        self.notebook.add(angles_tab, text="Joint Angles")
        self.notebook.add(debug_tab, text="Debug")

        angles_tab.columnconfigure(0, weight=1)
        angles_tab.rowconfigure(0, weight=1)
        debug_tab.columnconfigure(0, weight=1)
        debug_tab.rowconfigure(0, weight=1)

        # Joint angles display
        self._create_angles_display(angles_tab)
        self._create_debug_display(debug_tab)

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

        self.cal_min_button = ttk.Button(control_frame, text="Calibrate MIN",
                                         command=self.calibrate_min, width=20,
                                         state=tk.DISABLED)
        self.cal_min_button.grid(row=0, column=3, padx=5)

        self.cal_max_button = ttk.Button(control_frame, text="Calibrate MAX",
                                         command=self.calibrate_max, width=20,
                                         state=tk.DISABLED)
        self.cal_max_button.grid(row=0, column=4, padx=5)

        # Reset calibration
        self.reset_cal_button = ttk.Button(control_frame, text="Reset Calibration",
                                           command=self.reset_calibration, width=20)
        self.reset_cal_button.grid(row=0, column=5, padx=5)

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
        self.tree.heading('raw_deg', text='Sent (°)')
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
        debug_frame = ttk.LabelFrame(parent, text="Calculation Debug", padding="10")
        debug_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        debug_frame.columnconfigure(1, weight=1)

        labels = [
            ("Wrist:", "wrist"),
            ("Index MCP:", "index_mcp"),
            ("Pinky MCP:", "pinky_mcp"),
            ("Middle MCP:", "middle_mcp"),
            ("Middle PIP:", "middle_pip"),
            ("Plane Normal (index_mcp - pinky_mcp):", "plane_normal"),
            ("Vec1 (middle_mcp - wrist):", "vec1"),
            ("Vec2 (middle_pip - middle_mcp):", "vec2"),
            ("Vec1 Projected:", "vec1_proj"),
            ("Vec2 Projected:", "vec2_proj"),
            ("Vec1 Normalized:", "vec1_norm"),
            ("Vec2 Normalized:", "vec2_norm"),
            ("Angle (vec1_norm vs vec2_norm):", "angle_rad"),
        ]

        for row, (label, key) in enumerate(labels):
            ttk.Label(debug_frame, text=label).grid(
                row=row, column=0, sticky=tk.W, padx=(0, 10), pady=(0, 6)
            )
            ttk.Label(debug_frame, textvariable=self.debug_vars[key], font=('Arial', 10, 'bold')).grid(
                row=row, column=1, sticky=tk.W, pady=(0, 6)
            )

    def _format_vector(self, v: Tuple[float, float, float]) -> str:
        return f"({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})"

    def update_debug_display(self):
        """Update debug tab values."""
        if not self.debug_data:
            return

        vector_keys = [
            "wrist",
            "index_mcp",
            "pinky_mcp",
            "middle_mcp",
            "middle_pip",
            "plane_normal",
            "vec1",
            "vec2",
            "vec1_proj",
            "vec2_proj",
            "vec1_norm",
            "vec2_norm",
        ]
        for key in vector_keys:
            self.debug_vars[key].set(
                self._format_vector(self.debug_data.get(key, (0.0, 0.0, 0.0)))
            )

        angle_rad = float(self.debug_data.get("angle_rad", 0.0))
        self.debug_vars["angle_rad"].set(
            f"{angle_rad:.4f} rad ({math.degrees(angle_rad):.2f}°)"
        )

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
        """Enable all motors."""
        for motor_name in self.calibration.motor_names:
            self.motor_enabled[motor_name] = True
        self._update_tree_enabled_display()

        # Update status to indicate tracking resumed
        if self.running:
            self.status_label.config(text="Status: Running - MediaPipe tracking resumed")

    def deactivate_all_joints(self):
        """Disable all motors."""
        for motor_name in self.calibration.motor_names:
            self.motor_enabled[motor_name] = False
        self._update_tree_enabled_display()

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

            # Received aggregated angle for this motor
            recv_angle = self.current_angles.get(motor_name, 0.0)
            recv_deg = math.degrees(recv_angle)

            # Motor to angle: 255 -> 0, 0 -> PI
            sent_angle_rad = (1.0 - motor_value / 255.0) * math.pi
            sent_deg = math.degrees(sent_angle_rad)

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
                f"{sent_deg:.1f}",
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

    def start_controller(self):
        """Start the hand controller."""
        if self.running:
            return

        # Start control thread
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

        # Update UI immediately to indicate attempt
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.cal_min_button.config(state=tk.NORMAL)
        self.cal_max_button.config(state=tk.NORMAL)
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
        self.status_label.config(text="Status: Stopped")

    def calibrate_min(self):
        """Calibrate minimum values with current angles."""
        if self.current_angles:
            self.calibration.calibrate_min(self.current_angles)
            self.status_label.config(text="Status: MIN calibration captured!")
            self.update_angles_display()
            self.root.after(2000, lambda: self.status_label.config(
                text="Status: Running - Listening on tcp://localhost:5557"
            ))

    def calibrate_max(self):
        """Calibrate maximum values with current angles."""
        if self.current_angles:
            self.calibration.calibrate_max(self.current_angles)
            self.status_label.config(text="Status: MAX calibration captured!")
            self.update_angles_display()
            self.root.after(2000, lambda: self.status_label.config(
                text="Status: Running - Listening on tcp://localhost:5557"
            ))

    def reset_calibration(self):
        """Reset calibration to defaults."""
        self.calibration = CalibrationData()
        self.status_label.config(text="Status: Calibration reset to defaults")
        self.update_angles_display()
        self.root.after(2000, lambda: self.status_label.config(
            text="Status: Ready" if not self.running else
                 "Status: Running - Listening on tcp://localhost:5557"
        ))

    def _merge_poses(self, new_pose: List[int]) -> List[int]:
        """
        Merge new pose with last sent pose based on enabled motors.

        For enabled motors, use values from new_pose.
        For disabled motors, keep values from last_sent_pose.

        Args:
            new_pose: The newly calculated pose (20 motor values)

        Returns:
            Merged pose with enabled motors updated, disabled motors preserved
        """
        # Start with a copy of the last sent pose
        merged_pose = self.last_sent_pose.copy()

        # Update pose index for each enabled motor (1:1 mapping)
        for motor_name in self.calibration.motor_names:
            if self.motor_enabled.get(motor_name, True):
                # This motor is enabled, update its pose value
                motor_idx = self._get_pose_index_for_motor(motor_name)
                merged_pose[motor_idx] = new_pose[motor_idx]

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
            hand = LinkerHandApi(hand_type="left", hand_joint="L20")
            hand.set_speed([255, 255, 255, 255, 255])

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

                    try:
                        self.debug_data = calculate_middle_mcp_flexion_debug(landmarks)
                        self.root.after(0, self.update_debug_display)
                    except Exception:
                        pass

                    # Convert to motor angles
                    motor_angles = self._calculate_motor_angles(raw_angles)

                    # Store for display and calibration
                    self.current_angles = motor_angles

                    # Calculate new pose using calibrated values
                    new_pose = [0] * 20
                    for motor_name in self.calibration.motor_names:
                        angle = motor_angles.get(motor_name, 0.0)
                        motor_val = self.calibration.get_motor_value(motor_name, angle)
                        motor_idx = self._get_pose_index_for_motor(motor_name)
                        new_pose[motor_idx] = motor_val

                    # Merge with last sent pose based on enabled joints
                    self.current_pose = self._merge_poses(new_pose)

                    # Update last sent pose
                    self.last_sent_pose = self.current_pose.copy()

                    # Update display (in main thread)
                    self.root.after(0, self.update_angles_display)

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
