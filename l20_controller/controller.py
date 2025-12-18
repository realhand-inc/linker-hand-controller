import math
import sys
import time
import signal
from typing import Dict
import os

from LinkerHand.linker_hand_api import LinkerHandApi
from .network import ZmqReceiver, parse_landmarks
from .kinematics import calculate_all_joint_angles, angles_to_l20_pose

class L20Controller:
    def __init__(self, zmq_address: str = "tcp://localhost:5557"):
        self.receiver = ZmqReceiver(address=zmq_address)
        self.hand = LinkerHandApi(hand_type="left", hand_joint="L20")
        self.running = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        def handle_sigint(signum, frame):
            self.running = False
        signal.signal(signal.SIGINT, handle_sigint)

    def start(self):
        print("L20 Controller started - listening on tcp://localhost:5557")
        print("Press Ctrl+C to stop\n")
        
        # Initialize hand speed
        self.hand.set_speed([255, 255, 255, 255, 255])
        
        self.running = True
        try:
            while self.running:
                self.step()
                time.sleep(0.02)  # ~50Hz update rate
        finally:
            self.stop()

    def step(self):
        # Receive landmarks
        message = self.receiver.receive()
        if not message:
            return

        # Process landmarks
        try:
            landmarks = parse_landmarks(message)
            joint_angles = calculate_all_joint_angles(landmarks)
            pose = angles_to_l20_pose(joint_angles)
        except Exception as exc:
            print(f"\rParse error: {exc}                    ", end="", flush=True)
            return

        # Display status
        angle_summary = self._format_angle_summary(joint_angles)
        print(f"\r{angle_summary}", end="", flush=True)

        # Send to hand
        try:
            self.hand.finger_move(pose=pose)
        except Exception as exc:
            print(f"\nHand control error: {exc}")

    def stop(self):
        print("\nStopping L20 Controller...")
        self.hand.close_can()
        self.receiver.close()
        print("L20 Controller stopped.")

    def _format_angle_summary(self, angles: Dict[str, float]) -> str:
        """Format joint angles for terminal display."""
        thumb_avg = (angles.get('thumb_mcp', 0) + angles.get('thumb_ip', 0)) / 2.0
        index_avg = (angles.get('index_mcp', 0) + angles.get('index_pip', 0) + angles.get('index_dip', 0)) / 3.0
        middle_avg = (angles.get('middle_mcp', 0) + angles.get('middle_pip', 0) + angles.get('middle_dip', 0)) / 3.0
        ring_avg = (angles.get('ring_mcp', 0) + angles.get('ring_pip', 0) + angles.get('ring_dip', 0)) / 3.0
        pinky_avg = (angles.get('pinky_mcp', 0) + angles.get('pinky_pip', 0) + angles.get('pinky_dip', 0)) / 3.0

        return (f"T:{math.degrees(thumb_avg):5.1f}° "
                f"I:{math.degrees(index_avg):5.1f}° "
                f"M:{math.degrees(middle_avg):5.1f}° "
                f"R:{math.degrees(ring_avg):5.1f}° "
                f"P:{math.degrees(pinky_avg):5.1f}°")
