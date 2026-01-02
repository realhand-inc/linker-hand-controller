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

        # Shared calibration between both hands
        self.calibration = CalibrationData()

        # Left hand state
        self.left_active: bool = False
        self.left_can_interface: str = "can0"
        self._left_hand_instance: Optional[LinkerHandApi] = None
        self.left_current_angles: Dict[str, float] = {}
        self.left_current_pose: List[int] = [255] * 20
        self.left_last_sent_pose: List[int] = [255] * 20
        self.left_motor_enabled: Dict[str, bool] = {
            name: True for name in self.calibration.motor_names
        }

        # Right hand state
        self.right_active: bool = False
        self.right_can_interface: str = "can0"
        self._right_hand_instance: Optional[LinkerHandApi] = None
        self.right_current_angles: Dict[str, float] = {}
        self.right_current_pose: List[int] = [255] * 20
        self.right_last_sent_pose: List[int] = [255] * 20
        self.right_motor_enabled: Dict[str, bool] = {
            name: True for name in self.calibration.motor_names
        }

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

        left_angles_tab = ttk.Frame(self.notebook)
        right_angles_tab = ttk.Frame(self.notebook)
        debug_tab = ttk.Frame(self.notebook)

        self.notebook.add(left_angles_tab, text="Left Hand Angles")
        self.notebook.add(right_angles_tab, text="Right Hand Angles")
        self.notebook.add(debug_tab, text="Debug")

        left_angles_tab.columnconfigure(0, weight=1)
        left_angles_tab.rowconfigure(0, weight=1)
        right_angles_tab.columnconfigure(0, weight=1)
        right_angles_tab.rowconfigure(0, weight=1)
        debug_tab.columnconfigure(0, weight=1)
        debug_tab.rowconfigure(0, weight=1)

        # Joint angles display for each hand
        self._create_angles_display(left_angles_tab, hand_type="left")
        self._create_angles_display(right_angles_tab, hand_type="right")
        self._create_debug_display(debug_tab)

        # Status bar
        self.status_label = ttk.Label(main_frame, text="Status: Ready",
                                      relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

    def _create_control_panel(self, parent: ttk.Frame):
        """Create control buttons panel with dual hand activation."""
        control_frame = ttk.LabelFrame(parent, text="Control", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Left Hand Section
        ttk.Label(control_frame, text="Left Hand:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, padx=5, sticky=tk.W
        )
        ttk.Label(control_frame, text="CAN:").grid(row=0, column=1, padx=(0, 2))
        self.left_can_entry = ttk.Entry(control_frame, width=10)
        self.left_can_entry.insert(0, "can0")
        self.left_can_entry.grid(row=0, column=2, padx=(0, 5))

        self.left_activate_button = ttk.Button(
            control_frame, text="Activate Left",
            command=self.activate_left_hand, width=15
        )
        self.left_activate_button.grid(row=0, column=3, padx=5)

        self.left_deactivate_button = ttk.Button(
            control_frame, text="Deactivate Left",
            command=self.deactivate_left_hand, width=15,
            state=tk.DISABLED
        )
        self.left_deactivate_button.grid(row=0, column=4, padx=5)

        # Right Hand Section
        ttk.Label(control_frame, text="Right Hand:", font=('Arial', 10, 'bold')).grid(
            row=1, column=0, padx=5, sticky=tk.W
        )
        ttk.Label(control_frame, text="CAN:").grid(row=1, column=1, padx=(0, 2))
        self.right_can_entry = ttk.Entry(control_frame, width=10)
        self.right_can_entry.insert(0, "can0")
        self.right_can_entry.grid(row=1, column=2, padx=(0, 5))

        self.right_activate_button = ttk.Button(
            control_frame, text="Activate Right",
            command=self.activate_right_hand, width=15
        )
        self.right_activate_button.grid(row=1, column=3, padx=5)

        self.right_deactivate_button = ttk.Button(
            control_frame, text="Deactivate Right",
            command=self.deactivate_right_hand, width=15,
            state=tk.DISABLED
        )
        self.right_deactivate_button.grid(row=1, column=4, padx=5)

        # Separator
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(
            row=2, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=10
        )

        # Controller Section
        ttk.Label(control_frame, text="Controller:", font=('Arial', 10, 'bold')).grid(
            row=3, column=0, padx=5, sticky=tk.W
        )

        self.start_button = ttk.Button(
            control_frame, text="Start Controller",
            command=self.start_controller, width=20
        )
        self.start_button.grid(row=3, column=1, columnspan=2, padx=5)

        self.stop_button = ttk.Button(
            control_frame, text="Stop Controller",
            command=self.stop_controller, width=20,
            state=tk.DISABLED
        )
        self.stop_button.grid(row=3, column=3, padx=5)

        # Calibration Section
        ttk.Separator(control_frame, orient=tk.VERTICAL).grid(
            row=3, column=4, rowspan=2, sticky=(tk.N, tk.S), padx=10
        )

        self.cal_min_button = ttk.Button(
            control_frame, text="Calibrate MIN (Shared)",
            command=self.calibrate_min, width=22,
            state=tk.DISABLED
        )
        self.cal_min_button.grid(row=3, column=5, padx=5)

        self.cal_max_button = ttk.Button(
            control_frame, text="Calibrate MAX (Shared)",
            command=self.calibrate_max, width=22,
            state=tk.DISABLED
        )
        self.cal_max_button.grid(row=4, column=5, padx=5, pady=(5, 0))

        self.reset_cal_button = ttk.Button(
            control_frame, text="Reset Calibration",
            command=self.reset_calibration, width=22
        )
        self.reset_cal_button.grid(row=3, column=6, rowspan=2, padx=5)

    def _create_angles_display(self, parent: ttk.Frame, hand_type: str):
        """Create joint angles display table for a specific hand."""
        display_frame = ttk.LabelFrame(
            parent,
            text=f"{hand_type.title()} Hand Joint Angles",
            padding="10"
        )
        display_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)

        # Create style for taller rows
        style = ttk.Style()
        style.configure("Treeview", rowheight=28, font=('Arial', 10))
        style.configure("Treeview.Heading", font=('Arial', 10, 'bold'))

        # Create Treeview
        columns = ('enabled', 'recv_deg', 'raw_deg', 'cal_min', 'cal_max', 'remapped_deg', 'motor', 'goto_min', 'goto_max')
        tree = ttk.Treeview(display_frame, columns=columns, height=22)

        # Store tree reference based on hand type
        if hand_type == "left":
            self.left_tree = tree
        else:
            self.right_tree = tree

        # Define headings
        tree.heading('#0', text='Joint')
        tree.heading('enabled', text='Enabled')
        tree.heading('recv_deg', text='Recv (°)')
        tree.heading('raw_deg', text='Sent (°)')
        tree.heading('cal_min', text='Min (°)')
        tree.heading('cal_max', text='Max (°)')
        tree.heading('remapped_deg', text='Map (°)')
        tree.heading('motor', text='Motor')
        tree.heading('goto_min', text='Go Min')
        tree.heading('goto_max', text='Go Max')

        # Define column widths - increased for better visibility
        tree.column('#0', width=140, anchor=tk.W)
        tree.column('enabled', width=60, anchor=tk.CENTER)
        tree.column('recv_deg', width=80, anchor=tk.CENTER)
        tree.column('raw_deg', width=80, anchor=tk.CENTER)
        tree.column('cal_min', width=75, anchor=tk.CENTER)
        tree.column('cal_max', width=75, anchor=tk.CENTER)
        tree.column('remapped_deg', width=75, anchor=tk.CENTER)
        tree.column('motor', width=75, anchor=tk.CENTER)
        tree.column('goto_min', width=75, anchor=tk.CENTER)
        tree.column('goto_max', width=75, anchor=tk.CENTER)

        # Scrollbar
        scrollbar = ttk.Scrollbar(display_frame, orient=tk.VERTICAL,
                                  command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Add Activate/Deactivate All buttons
        button_frame = ttk.Frame(display_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0), sticky=tk.W)

        activate_all_button = ttk.Button(
            button_frame, text="Activate All Joints",
            command=lambda: self.activate_all_joints(hand_type), width=20
        )
        activate_all_button.grid(row=0, column=0, padx=5)

        deactivate_all_button = ttk.Button(
            button_frame, text="Deactivate All Joints",
            command=lambda: self.deactivate_all_joints(hand_type), width=20
        )
        deactivate_all_button.grid(row=0, column=1, padx=5)

        # Add Preset Gestures label and buttons
        preset_label = ttk.Label(button_frame, text="Preset Gestures:", font=('Arial', 10, 'bold'))
        preset_label.grid(row=1, column=0, columnspan=2, pady=(10, 5), sticky=tk.W)

        preset_fist_button = ttk.Button(
            button_frame, text="握拳 (Fist)",
            command=lambda: self.apply_preset("握拳", hand_type), width=15
        )
        preset_fist_button.grid(row=2, column=0, padx=5, pady=2)

        preset_open_button = ttk.Button(
            button_frame, text="张开 (Open)",
            command=lambda: self.apply_preset("张开", hand_type), width=15
        )
        preset_open_button.grid(row=2, column=1, padx=5, pady=2)

        preset_ok_button = ttk.Button(
            button_frame, text="OK",
            command=lambda: self.apply_preset("OK", hand_type), width=15
        )
        preset_ok_button.grid(row=3, column=0, padx=5, pady=2)

        preset_thumbsup_button = ttk.Button(
            button_frame, text="点赞 (Thumbs Up)",
            command=lambda: self.apply_preset("点赞", hand_type), width=15
        )
        preset_thumbsup_button.grid(row=3, column=1, padx=5, pady=2)

        # Populate joint names
        self._populate_joint_tree(tree, hand_type)

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

    def _populate_joint_tree(self, tree: ttk.Treeview, hand_type: str):
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
            parent = tree.insert('', tk.END, text=finger_name,
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

                # Namespace tree item IDs with hand type
                tree.insert(parent, tk.END, iid=f"{hand_type}_{motor_name}", text=display_name,
                                values=('\u2713', '0.0', '0.0', '0.0', '180.0', '0.0', '255', '[MIN]', '[MAX]'))

            # Expand all sections by default
            tree.item(parent, open=True)

        # Tag configuration for styling
        tree.tag_configure('parent', font=('Arial', 11, 'bold'), foreground='#0066cc')

        # Bind click event for toggling enabled state with hand type
        tree.bind('<Button-1>', lambda event: self._on_tree_click(event, hand_type, tree))

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

    def _go_to_motor_limit(self, motor_name: str, is_min: bool, hand_type: str):
        """
        Move a specific motor to limit (0 or 255) for specific hand.

        Args:
            motor_name: Name of the motor to move
            is_min: True to go to motor 0 (fully flexed), False to go to motor 255 (fully extended)
            hand_type: "left" or "right"
        """
        # Get hand-specific state
        active = self.left_active if hand_type == "left" else self.right_active
        hand_instance = self._left_hand_instance if hand_type == "left" else self._right_hand_instance

        if not active or not hand_instance:
            self.status_label.config(
                text=f"Status: Please activate {hand_type} hand before moving motors"
            )
            return

        # Set motor value to 0 (min/flexed) or 255 (max/extended)
        motor_value = 0 if is_min else 255

        # Get pose index for this motor (1:1 mapping)
        motor_idx = self._get_pose_index_for_motor(motor_name)

        # Update the motor position in hand-specific pose
        if hand_type == "left":
            self.left_last_sent_pose[motor_idx] = motor_value
            self.left_current_pose[motor_idx] = motor_value
            pose_to_send = self.left_last_sent_pose
        else:
            self.right_last_sent_pose[motor_idx] = motor_value
            self.right_current_pose[motor_idx] = motor_value
            pose_to_send = self.right_last_sent_pose

        # Send the updated pose
        try:
            hand_instance.finger_move(pose=pose_to_send)
            limit_type = "MIN (0)" if is_min else "MAX (255)"
            self.status_label.config(
                text=f"Status: Moved {hand_type} hand {motor_name} (motor {motor_idx}) to {limit_type}"
            )
            # Update display immediately
            self.update_angles_display(hand_type)
        except Exception as e:
            print(f"Error moving {hand_type} hand motor: {e}")
            self.status_label.config(text=f"Status: Error moving {hand_type} hand motor - {e}")

    def _get_pose_index_for_motor(self, motor_name: str) -> int:
        """Get pose index for a motor (1:1 mapping)."""
        # Direct 1:1 mapping - motor name index = pose index
        return self.calibration.motor_names.index(motor_name)

    def _on_tree_click(self, event, hand_type: str, tree: ttk.Treeview):
        """Handle click events on the tree to toggle enabled state or go to min/max."""
        # Identify which column was clicked
        column = tree.identify_column(event.x)

        # Identify which row was clicked
        row_id = tree.identify_row(event.y)
        if not row_id:
            return

        # Get hand-specific state
        motor_enabled = self.left_motor_enabled if hand_type == "left" else self.right_motor_enabled

        # Extract motor name from row_id (format: "left_thumb_base" or "right_index_tip")
        if row_id.startswith(f"{hand_type}_"):
            motor_name = row_id[len(f"{hand_type}_"):]
        else:
            motor_name = None

        # Handle "Go to Min" column (#8)
        if column == '#8':
            if motor_name and motor_name in self.calibration.motor_names:
                # Individual motor - go to min
                self._go_to_motor_limit(motor_name, is_min=True, hand_type=hand_type)
            else:
                # Parent row (finger group) - go to min for all motors
                finger_name = tree.item(row_id)['text']
                motors = self._get_finger_motors(finger_name)
                for motor in motors:
                    self._go_to_motor_limit(motor, is_min=True, hand_type=hand_type)
            return

        # Handle "Go to Max" column (#9)
        if column == '#9':
            if motor_name and motor_name in self.calibration.motor_names:
                # Individual motor - go to max
                self._go_to_motor_limit(motor_name, is_min=False, hand_type=hand_type)
            else:
                # Parent row (finger group) - go to max for all motors
                finger_name = tree.item(row_id)['text']
                motors = self._get_finger_motors(finger_name)
                for motor in motors:
                    self._go_to_motor_limit(motor, is_min=False, hand_type=hand_type)
            return

        # Handle enabled/disabled toggle (#1)
        if column != '#1':
            return

        # Check if it's a parent (finger group) or child (individual motor)
        if motor_name and motor_name in self.calibration.motor_names:
            # Individual motor
            motor_enabled[motor_name] = not motor_enabled[motor_name]
        else:
            # Parent row (finger group)
            finger_name = tree.item(row_id)['text']
            motors = self._get_finger_motors(finger_name)
            # Toggle all children - if any are disabled, enable all; otherwise disable all
            any_disabled = any(not motor_enabled.get(m, True) for m in motors)
            new_state = True if any_disabled else False
            for motor in motors:
                motor_enabled[motor] = new_state

        # Update the display
        self._update_tree_enabled_display(tree, hand_type)

    def _update_tree_enabled_display(self, tree: ttk.Treeview, hand_type: str):
        """Refresh the enabled column display for all items."""
        # Get hand-specific state
        motor_enabled = self.left_motor_enabled if hand_type == "left" else self.right_motor_enabled

        # Update individual motors
        for motor_name in self.calibration.motor_names:
            enabled = motor_enabled.get(motor_name, True)
            symbol = '\u2713' if enabled else '\u2717'  # ✓ or ✗
            try:
                current_values = list(tree.item(f"{hand_type}_{motor_name}")['values'])
                current_values[0] = symbol
                tree.item(f"{hand_type}_{motor_name}", values=current_values)
            except tk.TclError:
                pass

        # Update parent rows
        finger_groups = ["Thumb", "Index", "Middle", "Ring", "Pinky", "Reserved"]
        for item in tree.get_children():
            finger_name = tree.item(item)['text']
            if finger_name in finger_groups:
                motors = self._get_finger_motors(finger_name)
                enabled_count = sum(1 for m in motors if motor_enabled.get(m, True))
                if enabled_count == len(motors):
                    symbol = '\u2713'  # All enabled
                elif enabled_count == 0:
                    symbol = '\u2717'  # All disabled
                else:
                    symbol = '\u2212'  # Mixed (−)

                current_values = list(tree.item(item)['values'])
                current_values[0] = symbol
                try:
                    tree.item(item, values=current_values)
                except tk.TclError:
                    pass

    def activate_all_joints(self, hand_type: str):
        """Enable all motors for specific hand."""
        motor_enabled = self.left_motor_enabled if hand_type == "left" else self.right_motor_enabled
        tree = self.left_tree if hand_type == "left" else self.right_tree

        for motor_name in self.calibration.motor_names:
            motor_enabled[motor_name] = True
        self._update_tree_enabled_display(tree, hand_type)

        # Update status to indicate tracking resumed
        if self.running:
            self.status_label.config(text="Status: Running - MediaPipe tracking resumed")

    def deactivate_all_joints(self, hand_type: str):
        """Disable all motors for specific hand."""
        motor_enabled = self.left_motor_enabled if hand_type == "left" else self.right_motor_enabled
        tree = self.left_tree if hand_type == "left" else self.right_tree

        for motor_name in self.calibration.motor_names:
            motor_enabled[motor_name] = False
        self._update_tree_enabled_display(tree, hand_type)

    def apply_preset(self, gesture_name: str, hand_type: str):
        """Apply a preset gesture pose to specific hand."""
        if gesture_name not in self.preset_poses:
            return

        # Get hand-specific state
        active = self.left_active if hand_type == "left" else self.right_active
        hand_instance = self._left_hand_instance if hand_type == "left" else self._right_hand_instance

        if not active or not hand_instance:
            self.status_label.config(
                text=f"Status: Error - {hand_type.title()} hand not activated"
            )
            return

        preset_pose = self.preset_poses[gesture_name]

        # Send preset pose to hand
        try:
            hand_instance.finger_move(pose=preset_pose)
        except Exception as e:
            print(f"Error sending preset to {hand_type} hand: {e}")
            return

        # Update last sent pose and current pose for this hand
        if hand_type == "left":
            self.left_last_sent_pose = preset_pose.copy()
            self.left_current_pose = preset_pose.copy()
        else:
            self.right_last_sent_pose = preset_pose.copy()
            self.right_current_pose = preset_pose.copy()

        # Deactivate all joints to pause MediaPipe tracking
        self.deactivate_all_joints(hand_type)

        # Update display immediately
        self.update_angles_display(hand_type)

        # Update status
        self.status_label.config(
            text=f"Status: Preset '{gesture_name}' applied to {hand_type} hand - Click 'Activate All' to resume tracking"
        )

    def update_angles_display(self, hand_type: str):
        """Update the angles display with current motor values for specific hand."""
        # Get hand-specific state
        if hand_type == "left":
            tree = self.left_tree
            current_pose = self.left_current_pose
            current_angles = self.left_current_angles
            motor_enabled = self.left_motor_enabled
        else:
            tree = self.right_tree
            current_pose = self.right_current_pose
            current_angles = self.right_current_angles
            motor_enabled = self.right_motor_enabled

        # Display motor values directly from current_pose (20 motors)
        for motor_name in self.calibration.motor_names:
            motor_idx = self._get_pose_index_for_motor(motor_name)
            motor_value = current_pose[motor_idx]

            cal_min = self.calibration.min_values[motor_name]
            cal_max = self.calibration.max_values[motor_name]

            # Received aggregated angle for this motor
            recv_angle = current_angles.get(motor_name, 0.0)
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
            enabled = motor_enabled.get(motor_name, True)
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
                tree.item(f"{hand_type}_{motor_name}", values=values)
            except tk.TclError:
                pass  # Motor not in tree yet

        # Update parent rows to show aggregated enabled status
        finger_groups = ["Thumb", "Index", "Middle", "Ring", "Pinky", "Reserved"]
        for item in tree.get_children():
            finger_name = tree.item(item)['text']
            if finger_name in finger_groups:
                motors = self._get_finger_motors(finger_name)
                enabled_count = sum(1 for m in motors if motor_enabled.get(m, True))
                if enabled_count == len(motors):
                    symbol = '\u2713'  # All enabled
                elif enabled_count == 0:
                    symbol = '\u2717'  # All disabled
                else:
                    symbol = '\u2212'  # Mixed (−)

                current_values = list(tree.item(item)['values'])
                if current_values:
                    current_values[0] = symbol
                    try:
                        tree.item(item, values=current_values)
                    except tk.TclError:
                        pass

    def activate_left_hand(self):
        """Activate left hand - initialize hardware connection."""
        if self.left_active:
            self.status_label.config(text="Status: Left hand already active")
            return

        can_interface = self.left_can_entry.get().strip()
        if not can_interface:
            self.status_label.config(text="Status: Error - Please specify CAN interface for left hand")
            return

        self.left_can_interface = can_interface

        try:
            self._left_hand_instance = LinkerHandApi(
                hand_type="left",
                hand_joint="L20",
                can=self.left_can_interface
            )
            self._left_hand_instance.set_speed([255, 255, 255, 255, 255])

            self.left_active = True
            self.left_activate_button.config(state=tk.DISABLED)
            self.left_deactivate_button.config(state=tk.NORMAL)
            self.left_can_entry.config(state=tk.DISABLED)

            self.status_label.config(text=f"Status: Left hand activated on {self.left_can_interface}")

        except Exception as e:
            self.status_label.config(text=f"Status: Error activating left hand - {str(e)}")
            self._left_hand_instance = None
            self.left_active = False

    def deactivate_left_hand(self):
        """Deactivate left hand - close hardware connection."""
        if not self.left_active:
            return

        if self._left_hand_instance:
            try:
                self._left_hand_instance.close_can()
            except Exception:
                pass
            self._left_hand_instance = None

        self.left_active = False
        self.left_activate_button.config(state=tk.NORMAL)
        self.left_deactivate_button.config(state=tk.DISABLED)
        self.left_can_entry.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Left hand deactivated")

    def activate_right_hand(self):
        """Activate right hand - initialize hardware connection."""
        if self.right_active:
            self.status_label.config(text="Status: Right hand already active")
            return

        can_interface = self.right_can_entry.get().strip()
        if not can_interface:
            self.status_label.config(text="Status: Error - Please specify CAN interface for right hand")
            return

        self.right_can_interface = can_interface

        try:
            self._right_hand_instance = LinkerHandApi(
                hand_type="right",
                hand_joint="L20",
                can=self.right_can_interface
            )
            self._right_hand_instance.set_speed([255, 255, 255, 255, 255])

            self.right_active = True
            self.right_activate_button.config(state=tk.DISABLED)
            self.right_deactivate_button.config(state=tk.NORMAL)
            self.right_can_entry.config(state=tk.DISABLED)

            self.status_label.config(text=f"Status: Right hand activated on {self.right_can_interface}")

        except Exception as e:
            self.status_label.config(text=f"Status: Error activating right hand - {str(e)}")
            self._right_hand_instance = None
            self.right_active = False

    def deactivate_right_hand(self):
        """Deactivate right hand - close hardware connection."""
        if not self.right_active:
            return

        if self._right_hand_instance:
            try:
                self._right_hand_instance.close_can()
            except Exception:
                pass
            self._right_hand_instance = None

        self.right_active = False
        self.right_activate_button.config(state=tk.NORMAL)
        self.right_deactivate_button.config(state=tk.DISABLED)
        self.right_can_entry.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Right hand deactivated")

    def start_controller(self):
        """Start the hand controller (ZMQ listener only)."""
        if self.running:
            return

        # Check if at least one hand is activated
        if not self.left_active and not self.right_active:
            self.status_label.config(
                text="Status: Error - Please activate at least one hand first"
            )
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
        """Calibrate minimum values (shared across both hands)."""
        combined_angles = {}

        if self.left_active and self.left_current_angles:
            combined_angles.update(self.left_current_angles)

        if self.right_active and self.right_current_angles:
            combined_angles.update(self.right_current_angles)

        if combined_angles:
            self.calibration.calibrate_min(combined_angles)
            self.status_label.config(text="Status: MIN calibration captured (shared)!")

            # Update both displays
            if self.left_active:
                self.root.after(0, lambda: self.update_angles_display("left"))
            if self.right_active:
                self.root.after(0, lambda: self.update_angles_display("right"))

            self.root.after(2000, lambda: self.status_label.config(
                text="Status: Running"
            ))

    def calibrate_max(self):
        """Calibrate maximum values (shared across both hands)."""
        combined_angles = {}

        if self.left_active and self.left_current_angles:
            combined_angles.update(self.left_current_angles)

        if self.right_active and self.right_current_angles:
            combined_angles.update(self.right_current_angles)

        if combined_angles:
            self.calibration.calibrate_max(combined_angles)
            self.status_label.config(text="Status: MAX calibration captured (shared)!")

            # Update both displays
            if self.left_active:
                self.root.after(0, lambda: self.update_angles_display("left"))
            if self.right_active:
                self.root.after(0, lambda: self.update_angles_display("right"))

            self.root.after(2000, lambda: self.status_label.config(
                text="Status: Running"
            ))

    def reset_calibration(self):
        """Reset calibration to defaults."""
        self.calibration = CalibrationData()
        self.status_label.config(text="Status: Calibration reset to defaults")

        # Update both displays
        if self.left_active:
            self.update_angles_display("left")
        if self.right_active:
            self.update_angles_display("right")

        self.root.after(2000, lambda: self.status_label.config(
            text="Status: Ready" if not self.running else "Status: Running"
        ))

    def _merge_poses(self, new_pose: List[int], last_sent_pose: List[int],
                     motor_enabled: Dict[str, bool]) -> List[int]:
        """
        Merge new pose with last sent pose based on enabled motors.

        For enabled motors, use values from new_pose.
        For disabled motors, keep values from last_sent_pose.

        Args:
            new_pose: The newly calculated pose (20 motor values)
            last_sent_pose: The last pose sent to hardware
            motor_enabled: Dictionary of enabled/disabled motors for this hand

        Returns:
            Merged pose with enabled motors updated, disabled motors preserved
        """
        # Start with a copy of the last sent pose
        merged_pose = last_sent_pose.copy()

        # Update pose index for each enabled motor (1:1 mapping)
        for motor_name in self.calibration.motor_names:
            if motor_enabled.get(motor_name, True):
                # This motor is enabled, update its pose value
                motor_idx = self._get_pose_index_for_motor(motor_name)
                merged_pose[motor_idx] = new_pose[motor_idx]

        return merged_pose

    def _control_loop(self):
        """Main control loop running in separate thread - processes both hands."""
        zmq_socket = None

        try:
            # Initialize ZMQ socket
            context = zmq.Context.instance()
            zmq_socket = context.socket(zmq.SUB)
            zmq_socket.connect("tcp://localhost:5557")
            zmq_socket.setsockopt(zmq.SUBSCRIBE, b"")
            zmq_socket.RCVTIMEO = 500

            # Update status to show active hands
            active_hands = []
            if self.left_active:
                active_hands.append("Left")
            if self.right_active:
                active_hands.append("Right")

            hands_str = " + ".join(active_hands)
            self.root.after(0, lambda: self.status_label.config(
                text=f"Status: Running - {hands_str} hand(s) active - Listening on tcp://localhost:5557"
            ))

            while self.running:
                # Receive landmarks
                try:
                    message = zmq_socket.recv_string()
                except zmq.Again:
                    continue
                except zmq.ZMQError:
                    continue

                # Parse and calculate angles for BOTH hands
                try:
                    # Parse both left and right landmarks
                    from l20_controller import parse_landmarks_dual
                    left_landmarks, right_landmarks = parse_landmarks_dual(message)

                    # === PROCESS LEFT HAND ===
                    if self.left_active and left_landmarks:
                        try:
                            raw_angles = calculate_all_joint_angles(left_landmarks)
                            motor_angles = self._calculate_motor_angles(raw_angles)

                            # Store for display and calibration
                            self.left_current_angles = motor_angles

                            # Calculate new pose using calibrated values
                            new_pose = [0] * 20
                            for motor_name in self.calibration.motor_names:
                                angle = motor_angles.get(motor_name, 0.0)
                                motor_val = self.calibration.get_motor_value(motor_name, angle)
                                motor_idx = self._get_pose_index_for_motor(motor_name)
                                new_pose[motor_idx] = motor_val

                            # Merge with last sent pose based on enabled joints
                            self.left_current_pose = self._merge_poses(
                                new_pose, self.left_last_sent_pose, self.left_motor_enabled
                            )

                            # Update last sent pose
                            self.left_last_sent_pose = self.left_current_pose.copy()

                            # Send to left hand hardware
                            if self._left_hand_instance:
                                self._left_hand_instance.finger_move(pose=self.left_current_pose)

                            # Update display (in main thread)
                            self.root.after(0, lambda: self.update_angles_display("left"))

                        except Exception as e:
                            print(f"Left hand processing error: {e}")

                    # === PROCESS RIGHT HAND ===
                    if self.right_active and right_landmarks:
                        try:
                            raw_angles = calculate_all_joint_angles(right_landmarks)
                            motor_angles = self._calculate_motor_angles(raw_angles)

                            # Store for display and calibration
                            self.right_current_angles = motor_angles

                            # Calculate new pose using calibrated values
                            new_pose = [0] * 20
                            for motor_name in self.calibration.motor_names:
                                angle = motor_angles.get(motor_name, 0.0)
                                motor_val = self.calibration.get_motor_value(motor_name, angle)
                                motor_idx = self._get_pose_index_for_motor(motor_name)
                                new_pose[motor_idx] = motor_val

                            # Merge with last sent pose based on enabled joints
                            self.right_current_pose = self._merge_poses(
                                new_pose, self.right_last_sent_pose, self.right_motor_enabled
                            )

                            # Update last sent pose
                            self.right_last_sent_pose = self.right_current_pose.copy()

                            # Send to right hand hardware
                            if self._right_hand_instance:
                                self._right_hand_instance.finger_move(pose=self.right_current_pose)

                            # Update display (in main thread)
                            self.root.after(0, lambda: self.update_angles_display("right"))

                        except Exception as e:
                            print(f"Right hand processing error: {e}")

                    # Update debug display (using left hand for now)
                    if left_landmarks:
                        try:
                            self.debug_data = calculate_middle_mcp_flexion_debug(left_landmarks)
                            self.root.after(0, self.update_debug_display)
                        except Exception:
                            pass

                except ValueError as e:
                    if "No hand landmarks detected" in str(e):
                        # Update status to indicate no hand seen
                        self.root.after(0, lambda: self.status_label.config(
                            text="Status: Connected - No hand detected"
                        ))
                        continue
                    else:
                        print(f"Parse error: {e}")
                        continue
                except Exception as e:
                    print(f"Parse error: {e}")
                    continue

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
            # Cleanup ZMQ socket (hands remain active)
            if zmq_socket:
                try:
                    zmq_socket.close()
                except Exception:
                    pass

    def on_closing(self):
        """Handle window close event."""
        # Stop controller first
        self.stop_controller()

        # Deactivate both hands (closes hardware connections)
        if self.left_active:
            self.deactivate_left_hand()

        if self.right_active:
            self.deactivate_right_hand()

        # Destroy window
        self.root.destroy()


def main():
    """Run the GUI application."""
    root = tk.Tk()
    app = L20ControllerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
