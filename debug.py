from typing import Sequence, Tuple, Dict

from l20_controller.kinematics import LandmarkIndex
from l20_controller.math_utils import (
    vector_subtract,
    vector_dot,
    palm_plane_normal,
)


def calculate_thumb_yaw_debug(
    landmarks: Sequence[Tuple[float, float, float]]
) -> Dict[str, object]:
    """
    Return distance of thumb IP to palm plane for thumb yaw debug.

    Outputs:
        thumb_ip, middle_mcp, wrist, index_mcp, pinky_mcp: landmark positions
        plane_normal: cross(middle_mcp - wrist, index_mcp - pinky_mcp)
        distance: absolute distance from thumb_ip to plane (unit: same as landmarks)
    """
    thumb_ip = landmarks[LandmarkIndex.THUMB_IP]
    middle_mcp = landmarks[LandmarkIndex.MIDDLE_MCP]
    wrist = landmarks[LandmarkIndex.WRIST]
    index_mcp = landmarks[LandmarkIndex.INDEX_MCP]
    pinky_mcp = landmarks[LandmarkIndex.PINKY_MCP]

    plane_normal_norm = palm_plane_normal(middle_mcp, wrist, index_mcp, pinky_mcp)
    thumb_vec = vector_subtract(thumb_ip, wrist)
    distance = abs(vector_dot(thumb_vec, plane_normal_norm))

    return {
        "thumb_ip": thumb_ip,
        "middle_mcp": middle_mcp,
        "wrist": wrist,
        "index_mcp": index_mcp,
        "pinky_mcp": pinky_mcp,
        "plane_normal": plane_normal_norm,
        "distance": distance,
    }


def calculate_middle_tip_debug(
    landmarks: Sequence[Tuple[float, float, float]]
) -> Dict[str, object]:
    """
    Return reference vectors and landmark positions for middle finger tip angle debug.

    Outputs:
        middle_mcp, middle_dip, middle_tip: landmark positions
        index_mcp, pinky_mcp: landmark positions for reference
        vec1: DIP → TIP vector
        vec2: MCP → DIP vector
    """
    middle_mcp = landmarks[LandmarkIndex.MIDDLE_MCP]
    middle_dip = landmarks[LandmarkIndex.MIDDLE_DIP]
    middle_tip = landmarks[LandmarkIndex.MIDDLE_TIP]
    index_mcp = landmarks[LandmarkIndex.INDEX_MCP]
    pinky_mcp = landmarks[LandmarkIndex.PINKY_MCP]

    # Reference vectors for tip angle calculation
    vec1 = vector_subtract(middle_tip, middle_dip)  # DIP → TIP
    vec2 = vector_subtract(middle_dip, middle_mcp)  # MCP → DIP

    return {
        "middle_mcp": middle_mcp,
        "middle_dip": middle_dip,
        "middle_tip": middle_tip,
        "index_mcp": index_mcp,
        "pinky_mcp": pinky_mcp,
        "vec1": vec1,
        "vec2": vec2,
    }
