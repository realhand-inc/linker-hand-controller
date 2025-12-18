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
        angles_to_l20_pose,
    )
except ImportError:
    print("Error: Could not import from l20_controller.py")
    print("Make sure l20_controller.py is in the same directory.")
    sys.exit(1)


class CalibrationData:
    """Stores min/max calibration values for each joint."""

    def __init__(self):
        self.joint_names = [
            'thumb_cmc_abd', 'thumb_cmc_flex', 'thumb_mcp', 'thumb_ip',
            'index_mcp', 'index_pip', 'index_dip',
            'middle_mcp', 'middle_pip', 'middle_dip',
            'ring_mcp', 'ring_pip', 'ring_dip',
            'pinky_mcp', 'pinky_pip', 'pinky_dip',
        ]

        # Initialize with default ranges (0 to π radians)
        self.min_values: Dict[str, float] = {name: 0.0 for name in self.joint_names}
        self.max_values: Dict[str, float] = {name: math.pi for name in self.joint_names}

        # Special ranges for thumb CMC joints
        self.max_values['thumb_cmc_abd'] = math.pi / 2
        self.max_values['thumb_cmc_flex'] = math.pi / 2

    def calibrate_min(self, current_angles: Dict[str, float]):
        """Set current angles as minimum calibration values."""
        for name in self.joint_names:
            if name in current_angles:
                self.min_values[name] = current_angles[name]

    def calibrate_max(self, current_angles: Dict[str, float]):
        """Set current angles as maximum calibration values."""
        for name in self.joint_names:
            if name in current_angles:
                self.max_values[name] = current_angles[name]

    def remap_angle(self, joint_name: str, angle: float) -> float:
        """
        Remap angle from calibrated range to [0, π] range.

        Formula: remapped = (angle - min) / (max - min) * π
        """
        min_val = self.min_values.get(joint_name, 0.0)
        max_val = self.max_values.get(joint_name, math.pi)

        # Avoid division by zero
        if abs(max_val - min_val) < 0.001:
            return angle

        # Clamp and remap
        clamped = max(min_val, min(max_val, angle))
        normalized = (clamped - min_val) / (max_val - min_val)

        # Remap to target range
        target_max = math.pi / 2 if 'cmc' in joint_name else math.pi
        return normalized * target_max

    def remap_all_angles(self, angles: Dict[str, float]) -> Dict[str, float]:
        """Remap all joint angles using calibration data."""
        remapped = {}
        for name, angle in angles.items():
            remapped[name] = self.remap_angle(name, angle)
        return remapped


class L20ControllerGUI:
    """GUI for L20 hand controller with calibration."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("L20 Hand Controller - Calibration Interface")
        self.root.geometry("1800x1200")

        self.calibration = CalibrationData()
        self.current_angles: Dict[str, float] = {}
        self.remapped_angles: Dict[str, float] = {}
        self.current_pose: List[int] = [255] * 20

        # Track which joints are enabled for sending to robot
        self.joint_enabled: Dict[str, bool] = {
            name: True for name in self.calibration.joint_names
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

        # Joint angles display
        self._create_angles_display(main_frame)

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
        display_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)

        # Create style for taller rows
        style = ttk.Style()
        style.configure("Treeview", rowheight=28, font=('Arial', 10))
        style.configure("Treeview.Heading", font=('Arial', 10, 'bold'))

        # Create Treeview
        columns = ('enabled', 'raw_deg', 'raw_rad', 'cal_min', 'cal_max', 'remapped_deg', 'motor')
        self.tree = ttk.Treeview(display_frame, columns=columns, height=22)

        # Define headings
        self.tree.heading('#0', text='Joint')
        self.tree.heading('enabled', text='Enabled')
        self.tree.heading('raw_deg', text='Raw (°)')
        self.tree.heading('raw_rad', text='Raw (rad)')
        self.tree.heading('cal_min', text='Min (°)')
        self.tree.heading('cal_max', text='Max (°)')
        self.tree.heading('remapped_deg', text='Remapped (°)')
        self.tree.heading('motor', text='Motor (0-255)')

        # Define column widths - increased for better visibility
        self.tree.column('#0', width=150, anchor=tk.W)
        self.tree.column('enabled', width=80, anchor=tk.CENTER)
        self.tree.column('raw_deg', width=100, anchor=tk.CENTER)
        self.tree.column('raw_rad', width=100, anchor=tk.CENTER)
        self.tree.column('cal_min', width=100, anchor=tk.CENTER)
        self.tree.column('cal_max', width=100, anchor=tk.CENTER)
        self.tree.column('remapped_deg', width=120, anchor=tk.CENTER)
        self.tree.column('motor', width=130, anchor=tk.CENTER)

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

    def _populate_joint_tree(self):
        """Populate the tree with joint names grouped by finger."""
        finger_groups = [
            ("Thumb", ['thumb_cmc_abd', 'thumb_cmc_flex', 'thumb_mcp', 'thumb_ip']),
            ("Index", ['index_mcp', 'index_pip', 'index_dip']),
            ("Middle", ['middle_mcp', 'middle_pip', 'middle_dip']),
            ("Ring", ['ring_mcp', 'ring_pip', 'ring_dip']),
            ("Pinky", ['pinky_mcp', 'pinky_pip', 'pinky_dip']),
        ]

        for finger_name, joints in finger_groups:
            # Insert parent (finger name)
            parent = self.tree.insert('', tk.END, text=finger_name,
                                      values=('\u2713', '', '', '', '', '', ''),
                                      tags=('parent',))

            # Insert children (joint names)
            for joint in joints:
                joint_display = joint.replace('_', ' ').upper()
                self.tree.insert(parent, tk.END, iid=joint, text=f"  {joint_display}",
                                values=('\u2713', '0.0', '0.00', '0.0', '180.0', '0.0', '255'))

            # Expand all sections by default
            self.tree.item(parent, open=True)

        # Tag configuration for styling
        self.tree.tag_configure('parent', font=('Arial', 11, 'bold'), foreground='#0066cc')

        # Bind click event for toggling enabled state
        self.tree.bind('<Button-1>', self._on_tree_click)

    def _get_finger_joints(self, finger_name: str) -> List[str]:
        """Get list of joint names for a given finger."""
        finger_map = {
            "Thumb": ['thumb_cmc_abd', 'thumb_cmc_flex', 'thumb_mcp', 'thumb_ip'],
            "Index": ['index_mcp', 'index_pip', 'index_dip'],
            "Middle": ['middle_mcp', 'middle_pip', 'middle_dip'],
            "Ring": ['ring_mcp', 'ring_pip', 'ring_dip'],
            "Pinky": ['pinky_mcp', 'pinky_pip', 'pinky_dip'],
        }
        return finger_map.get(finger_name, [])

    def _get_pose_indices_for_joint(self, joint_name: str) -> List[int]:
        """Get which pose indices are affected by this joint."""
        # Based on the mapping in kinematics.py angles_to_l20_pose
        mapping = {
            'thumb_cmc_abd': [5],
            'thumb_cmc_flex': [10],
            'thumb_mcp': [0, 15],
            'thumb_ip': [0, 15],
            'index_mcp': [1, 16],
            'index_pip': [1, 16],
            'index_dip': [1, 16],
            'middle_mcp': [2, 17],
            'middle_pip': [2, 17],
            'middle_dip': [2, 17],
            'ring_mcp': [3, 18],
            'ring_pip': [3, 18],
            'ring_dip': [3, 18],
            'pinky_mcp': [4, 19],
            'pinky_pip': [4, 19],
            'pinky_dip': [4, 19],
        }
        return mapping.get(joint_name, [])

    def _on_tree_click(self, event):
        """Handle click events on the tree to toggle enabled state."""
        # Identify which column was clicked
        column = self.tree.identify_column(event.x)
        if column != '#1':  # #1 is the first data column (enabled)
            return

        # Identify which row was clicked
        row_id = self.tree.identify_row(event.y)
        if not row_id:
            return

        # Check if it's a parent (finger group) or child (individual joint)
        if row_id in self.calibration.joint_names:
            # Individual joint
            self.joint_enabled[row_id] = not self.joint_enabled[row_id]
        else:
            # Parent row (finger group)
            joints = self._get_finger_joints(self.tree.item(row_id)['text'])
            # Toggle all children - if any are disabled, enable all; otherwise disable all
            any_disabled = any(not self.joint_enabled.get(j, True) for j in joints)
            new_state = True if any_disabled else False
            for joint in joints:
                self.joint_enabled[joint] = new_state

        # Update the display
        self._update_tree_enabled_display()

    def _update_tree_enabled_display(self):
        """Refresh the enabled column display for all items."""
        # Update individual joints
        for joint_name in self.calibration.joint_names:
            enabled = self.joint_enabled.get(joint_name, True)
            symbol = '\u2713' if enabled else '\u2717'  # ✓ or ✗
            current_values = list(self.tree.item(joint_name)['values'])
            current_values[0] = symbol
            try:
                self.tree.item(joint_name, values=current_values)
            except tk.TclError:
                pass

        # Update parent rows
        finger_groups = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        for item in self.tree.get_children():
            finger_name = self.tree.item(item)['text']
            if finger_name in finger_groups:
                joints = self._get_finger_joints(finger_name)
                enabled_count = sum(1 for j in joints if self.joint_enabled.get(j, True))
                if enabled_count == len(joints):
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
        """Enable all joints."""
        for joint_name in self.calibration.joint_names:
            self.joint_enabled[joint_name] = True
        self._update_tree_enabled_display()

        # Update status to indicate tracking resumed
        if self.running:
            self.status_label.config(text="Status: Running - MediaPipe tracking resumed")

    def deactivate_all_joints(self):
        """Disable all joints."""
        for joint_name in self.calibration.joint_names:
            self.joint_enabled[joint_name] = False
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

        # Update last sent pose
        self.last_sent_pose = preset_pose.copy()

        # Deactivate all joints to pause MediaPipe tracking
        self.deactivate_all_joints()

        # Update status
        self.status_label.config(text=f"Status: Preset '{gesture_name}' applied - Click 'Activate All' to resume tracking")

    def update_angles_display(self):
        """Update the angles display with current values."""
        for joint_name in self.calibration.joint_names:
            raw_angle = self.current_angles.get(joint_name, 0.0)
            remapped_angle = self.remapped_angles.get(joint_name, 0.0)

            cal_min = self.calibration.min_values[joint_name]
            cal_max = self.calibration.max_values[joint_name]

            # Calculate motor value
            max_angle = math.pi / 2 if 'cmc' in joint_name else math.pi
            motor_value = int((1.0 - remapped_angle / max_angle) * 255)
            motor_value = max(0, min(255, motor_value))

            # Get enabled status
            enabled = self.joint_enabled.get(joint_name, True)
            enabled_symbol = '\u2713' if enabled else '\u2717'  # ✓ or ✗

            values = (
                enabled_symbol,
                f"{math.degrees(raw_angle):.1f}",
                f"{raw_angle:.2f}",
                f"{math.degrees(cal_min):.1f}",
                f"{math.degrees(cal_max):.1f}",
                f"{math.degrees(remapped_angle):.1f}",
                f"{motor_value}"
            )

            try:
                self.tree.item(joint_name, values=values)
            except tk.TclError:
                pass  # Joint not in tree yet

        # Update parent rows to show aggregated enabled status
        finger_groups = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        for item in self.tree.get_children():
            finger_name = self.tree.item(item)['text']
            if finger_name in finger_groups:
                joints = self._get_finger_joints(finger_name)
                enabled_count = sum(1 for j in joints if self.joint_enabled.get(j, True))
                if enabled_count == len(joints):
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
            self.root.after(2000, lambda: self.status_label.config(
                text="Status: Running - Listening on tcp://localhost:5557"
            ))

    def calibrate_max(self):
        """Calibrate maximum values with current angles."""
        if self.current_angles:
            self.calibration.calibrate_max(self.current_angles)
            self.status_label.config(text="Status: MAX calibration captured!")
            self.root.after(2000, lambda: self.status_label.config(
                text="Status: Running - Listening on tcp://localhost:5557"
            ))

    def reset_calibration(self):
        """Reset calibration to defaults."""
        self.calibration = CalibrationData()
        self.status_label.config(text="Status: Calibration reset to defaults")
        self.root.after(2000, lambda: self.status_label.config(
            text="Status: Ready" if not self.running else
                 "Status: Running - Listening on tcp://localhost:5557"
        ))

    def _merge_poses(self, new_pose: List[int], remapped_angles: Dict[str, float]) -> List[int]:
        """
        Merge new pose with last sent pose based on enabled joints.

        For enabled joints, use values from new_pose.
        For disabled joints, keep values from last_sent_pose.

        Args:
            new_pose: The newly calculated pose (20 motor values)
            remapped_angles: The remapped joint angles used to calculate new_pose

        Returns:
            Merged pose with enabled joints updated, disabled joints preserved
        """
        # Start with a copy of the last sent pose
        merged_pose = self.last_sent_pose.copy()

        # Track which pose indices have been updated by enabled joints
        updated_indices = set()

        # Update pose indices for each enabled joint
        for joint_name in self.calibration.joint_names:
            if self.joint_enabled.get(joint_name, True):
                # This joint is enabled, update its corresponding pose indices
                indices = self._get_pose_indices_for_joint(joint_name)
                for idx in indices:
                    merged_pose[idx] = new_pose[idx]
                    updated_indices.add(idx)

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

                    # Store raw angles
                    self.current_angles = raw_angles

                    # Apply calibration remapping
                    self.remapped_angles = self.calibration.remap_all_angles(raw_angles)

                    # Convert to pose
                    new_pose = angles_to_l20_pose(self.remapped_angles)

                    # Merge with last sent pose based on enabled joints
                    self.current_pose = self._merge_poses(new_pose, self.remapped_angles)

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
