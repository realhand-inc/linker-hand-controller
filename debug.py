import math
from typing import Sequence, Tuple, Dict

from l20_controller.kinematics import LandmarkIndex
from l20_controller.math_utils import (
    vector_subtract,
    vector_normalize,
    project_vector_onto_plane,
    vector_dot,
)


def calculate_middle_mcp_flexion_debug(
    landmarks: Sequence[Tuple[float, float, float]]
) -> Dict[str, object]:
    """
    Return vectors and angle used for middle MCP flexion calculation.

    Outputs:
        wrist, index_mcp, pinky_mcp, middle_mcp, middle_pip: landmark positions
        plane_normal: index_mcp - pinky_mcp (normalized)
        vec1: middle_mcp - wrist
        vec2: middle_pip - middle_mcp
        vec1_proj: vec1 projected onto plane perpendicular to plane_normal
        vec2_proj: vec2 projected onto plane perpendicular to plane_normal
        vec1_norm: normalized vec1_proj
        vec2_norm: normalized vec2_proj
        angle_rad: angle between vec1_norm and vec2_norm
    """
    wrist = landmarks[LandmarkIndex.WRIST]
    index_mcp = landmarks[LandmarkIndex.INDEX_MCP]
    pinky_mcp = landmarks[LandmarkIndex.PINKY_MCP]
    middle_mcp = landmarks[LandmarkIndex.MIDDLE_MCP]
    middle_pip = landmarks[LandmarkIndex.MIDDLE_PIP]

    plane_normal_raw = vector_subtract(index_mcp, pinky_mcp)
    plane_normal = vector_normalize(plane_normal_raw)

    vec1 = vector_subtract(middle_mcp, wrist)
    vec2 = vector_subtract(middle_pip, middle_mcp)

    vec1_proj = project_vector_onto_plane(vec1, plane_normal)
    vec2_proj = project_vector_onto_plane(vec2, plane_normal)

    vec1_norm = vector_normalize(vec1_proj)
    vec2_norm = vector_normalize(vec2_proj)

    dot = max(-1.0, min(1.0, vector_dot(vec1_norm, vec2_norm)))
    angle_rad = math.acos(dot)

    return {
        "wrist": wrist,
        "index_mcp": index_mcp,
        "pinky_mcp": pinky_mcp,
        "middle_mcp": middle_mcp,
        "middle_pip": middle_pip,
        "plane_normal": plane_normal_raw,
        "vec1": vec1,
        "vec2": vec2,
        "vec1_proj": vec1_proj,
        "vec2_proj": vec2_proj,
        "vec1_norm": vec1_norm,
        "vec2_norm": vec2_norm,
        "angle_rad": angle_rad,
    }
