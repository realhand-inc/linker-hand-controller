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
        self.root.geometry("1000x800")

        self.calibration = CalibrationData()
        self.current_angles: Dict[str, float] = {}
        self.remapped_angles: Dict[str, float] = {}
        self.current_pose: List[int] = [255] * 20

        self.running = False
        self.hand: Optional[LinkerHandApi] = None
        self.zmq_socket: Optional[zmq.Socket] = None
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
        columns = ('raw_deg', 'raw_rad', 'cal_min', 'cal_max', 'remapped_deg', 'motor')
        self.tree = ttk.Treeview(display_frame, columns=columns, height=22)

        # Define headings
        self.tree.heading('#0', text='Joint')
        self.tree.heading('raw_deg', text='Raw (°)')
        self.tree.heading('raw_rad', text='Raw (rad)')
        self.tree.heading('cal_min', text='Min (°)')
        self.tree.heading('cal_max', text='Max (°)')
        self.tree.heading('remapped_deg', text='Remapped (°)')
        self.tree.heading('motor', text='Motor (0-255)')

        # Define column widths - increased for better visibility
        self.tree.column('#0', width=150, anchor=tk.W)
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
                                      values=('', '', '', '', '', ''),
                                      tags=('parent',))

            # Insert children (joint names)
            for joint in joints:
                joint_display = joint.replace('_', ' ').upper()
                self.tree.insert(parent, tk.END, iid=joint, text=f"  {joint_display}",
                                values=('0.0', '0.00', '0.0', '180.0', '0.0', '255'))

            # Expand all sections by default
            self.tree.item(parent, open=True)

        # Tag configuration for styling
        self.tree.tag_configure('parent', font=('Arial', 11, 'bold'), foreground='#0066cc')

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

            values = (
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

    def start_controller(self):
        """Start the hand controller."""
        if self.running:
            return

        try:
            # Initialize ZMQ socket
            context = zmq.Context.instance()
            self.zmq_socket = context.socket(zmq.SUB)
            self.zmq_socket.connect("tcp://localhost:5557")
            self.zmq_socket.setsockopt(zmq.SUBSCRIBE, b"")
            self.zmq_socket.RCVTIMEO = 500

            # Initialize hand
            self.hand = LinkerHandApi(hand_type="left", hand_joint="L20")
            self.hand.set_speed([255, 255, 255, 255, 255])

            # Start control thread
            self.running = True
            self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self.control_thread.start()

            # Update UI
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.cal_min_button.config(state=tk.NORMAL)
            self.cal_max_button.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Running - Listening on tcp://localhost:5557")

        except Exception as e:
            self.status_label.config(text=f"Status: Error - {e}")
            self.running = False

    def stop_controller(self):
        """Stop the hand controller."""
        self.running = False

        if self.control_thread:
            self.control_thread.join(timeout=2.0)

        if self.hand:
            try:
                self.hand.close_can()
            except Exception:  # noqa: S110
                pass

        if self.zmq_socket:
            self.zmq_socket.close()

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

    def _control_loop(self):
        """Main control loop running in separate thread."""
        while self.running:
            try:
                # Receive landmarks
                try:
                    message = self.zmq_socket.recv_string()
                except zmq.Again:
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
                    self.current_pose = angles_to_l20_pose(self.remapped_angles)

                    # Update display (in main thread)
                    self.root.after(0, self.update_angles_display)

                except Exception as e:  # noqa: BLE001
                    print(f"Parse error: {e}")
                    continue

                # Send to hand
                try:
                    if self.hand:
                        self.hand.finger_move(pose=self.current_pose)
                except Exception as e:  # noqa: BLE001
                    print(f"Hand control error: {e}")

                time.sleep(0.02)  # ~50Hz

            except Exception as e:  # noqa: BLE001
                print(f"Control loop error: {e}")
                time.sleep(0.1)

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
