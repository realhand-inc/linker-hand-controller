import json
from typing import List, Tuple
import zmq

class ZmqReceiver:
    def __init__(self, address: str = "tcp://localhost:5557", timeout_ms: int = 500):
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(address)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.RCVTIMEO = timeout_ms

    def receive(self) -> str | None:
        try:
            return self.socket.recv_string()
        except zmq.Again:
            return None

    def close(self):
        self.socket.close()


def parse_landmarks(message: str) -> List[Tuple[float, float, float]]:
    """
    Parse ZMQ JSON message into 21 hand landmarks.

    Expected format: {"left": [{x,y,z}, ...], "right": [{x,y,z}, ...]}
    """
    try:
        payload = json.loads(message)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    if not isinstance(payload, dict):
        raise ValueError("Expected dict with 'left'/'right' keys")

    landmarks = payload.get("left") or payload.get("right") or []
    if not landmarks:
        raise ValueError("No hand landmarks detected")

    coords: List[Tuple[float, float, float]] = []
    for entry in landmarks:
        if isinstance(entry, dict):
            coords.append((
                float(entry.get("x", 0.0)),
                float(entry.get("y", 0.0)),
                float(entry.get("z", 0.0))
            ))
        elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
            coords.append((float(entry[0]), float(entry[1]), float(entry[2])))

    if len(coords) < 21:
        raise ValueError(f"Expected 21 landmarks, got {len(coords)}")

    return coords[:21]
