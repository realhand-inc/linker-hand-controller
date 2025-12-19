from typing import Sequence, Tuple, Dict, List
import math
from .math_utils import (
    vector_subtract,
    vector_cross,
    vector_normalize,
    signed_angle_in_plane,
    project_vector_onto_plane,
    vector_dot
)

# MediaPipe hand landmark indices
class LandmarkIndex:
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# ============================================================================
# THUMB JOINT ANGLE CALCULATIONS
# ============================================================================

def calculate_thumb_cmc_abduction(landmarks: Sequence[Tuple[float, float, float]]) -> float:
    """
    Calculate thumb CMC (carpometacarpal) abduction/adduction angle.

    Projects onto the plane perpendicular to the palm (frontal plane).
    Measures angle between wrist→index_mcp and wrist→thumb_cmc.

    Returns: angle in radians [0, π/2], where 0 = adducted, π/2 = abducted
    """
    wrist = landmarks[LandmarkIndex.WRIST]
    thumb_cmc = landmarks[LandmarkIndex.THUMB_CMC]
    index_mcp = landmarks[LandmarkIndex.INDEX_MCP]
    middle_mcp = landmarks[LandmarkIndex.MIDDLE_MCP]

    # Define palm plane using wrist and two finger bases
    v_index = vector_subtract(index_mcp, wrist)
    v_middle = vector_subtract(middle_mcp, wrist)
    palm_normal = vector_normalize(vector_cross(v_index, v_middle))

    # Reference vector: wrist to index base
    ref_vector = vector_subtract(index_mcp, wrist)

    # Vector to measure: wrist to thumb CMC
    thumb_vector = vector_subtract(thumb_cmc, wrist)

    # Calculate angle in the frontal plane (perpendicular to palm)
    angle = signed_angle_in_plane(ref_vector, thumb_vector, palm_normal)

    return abs(angle)  # Return absolute value for abduction


def calculate_thumb_cmc_flexion(landmarks: Sequence[Tuple[float, float, float]]) -> float:
    """
    Calculate thumb CMC flexion/extension angle.

    Projects onto the sagittal plane (parallel to palm, perpendicular to abduction).
    Measures forward/backward motion of thumb base.

    Returns: angle in radians [0, π/2], where 0 = extended, π/2 = flexed
    """
    wrist = landmarks[LandmarkIndex.WRIST]
    thumb_cmc = landmarks[LandmarkIndex.THUMB_CMC]
    thumb_mcp = landmarks[LandmarkIndex.THUMB_MCP]
    index_mcp = landmarks[LandmarkIndex.INDEX_MCP]

    # Abduction axis (perpendicular to palm, pointing outward)
    middle_mcp = landmarks[LandmarkIndex.MIDDLE_MCP]
    v_index = vector_subtract(index_mcp, wrist)
    v_middle = vector_subtract(middle_mcp, wrist)
    abduction_axis = vector_normalize(vector_cross(v_index, v_middle))

    # Reference vector along the palm
    ref_vector = vector_subtract(index_mcp, wrist)

    # Thumb metacarpal vector
    thumb_vector = vector_subtract(thumb_mcp, thumb_cmc)

    # Project onto plane perpendicular to abduction axis
    angle = signed_angle_in_plane(ref_vector, thumb_vector, abduction_axis)

    return max(0.0, angle)  # Flexion is positive angle


def calculate_thumb_mcp_flexion(landmarks: Sequence[Tuple[float, float, float]]) -> float:
    """
    Calculate thumb MCP (metacarpophalangeal) joint flexion angle.

    Projects onto the plane defined by the thumb metacarpal bone axis.
    Measures bending at the first thumb knuckle.

    Returns: angle in radians [0, π], where 0 = straight, π = fully flexed
    """
    thumb_cmc = landmarks[LandmarkIndex.THUMB_CMC]
    thumb_mcp = landmarks[LandmarkIndex.THUMB_MCP]
    thumb_ip = landmarks[LandmarkIndex.THUMB_IP]
    thumb_tip = landmarks[LandmarkIndex.THUMB_TIP] # noqa: F841
    wrist = landmarks[LandmarkIndex.WRIST]
    index_mcp = landmarks[LandmarkIndex.INDEX_MCP]

    # Define rotation axis (perpendicular to thumb motion plane)
    # Use palm normal as the rotation axis
    middle_mcp = landmarks[LandmarkIndex.MIDDLE_MCP]
    v1 = vector_subtract(index_mcp, wrist)
    v2 = vector_subtract(middle_mcp, wrist)
    rotation_axis = vector_normalize(vector_cross(v1, v2))

    # Proximal segment (CMC to MCP)
    proximal = vector_subtract(thumb_mcp, thumb_cmc)

    # Distal segment (MCP to IP)
    distal = vector_subtract(thumb_ip, thumb_mcp)

    # Calculate angle in the plane perpendicular to rotation axis
    angle = signed_angle_in_plane(proximal, distal, rotation_axis)

    # Return supplementary angle for flexion (π - angle)
    return math.pi - max(0.0, min(math.pi, angle))


def calculate_thumb_ip_flexion(landmarks: Sequence[Tuple[float, float, float]]) -> float:
    """
    Calculate thumb IP (interphalangeal) joint flexion angle.

    Projects onto the plane of thumb flexion.
    Measures bending at the thumb tip joint.

    Returns: angle in radians [0, π], where 0 = straight, π = fully flexed
    """
    thumb_mcp = landmarks[LandmarkIndex.THUMB_MCP]
    thumb_ip = landmarks[LandmarkIndex.THUMB_IP]
    thumb_tip = landmarks[LandmarkIndex.THUMB_TIP]
    wrist = landmarks[LandmarkIndex.WRIST]
    index_mcp = landmarks[LandmarkIndex.INDEX_MCP]
    middle_mcp = landmarks[LandmarkIndex.MIDDLE_MCP]

    # Define rotation axis using palm normal
    v1 = vector_subtract(index_mcp, wrist)
    v2 = vector_subtract(middle_mcp, wrist)
    rotation_axis = vector_normalize(vector_cross(v1, v2))

    # Proximal segment (MCP to IP)
    proximal = vector_subtract(thumb_ip, thumb_mcp)

    # Distal segment (IP to TIP)
    distal = vector_subtract(thumb_tip, thumb_ip)

    # Calculate angle in the plane perpendicular to rotation axis
    angle = signed_angle_in_plane(proximal, distal, rotation_axis)

    # Return supplementary angle for flexion
    return math.pi - max(0.0, min(math.pi, angle))


# ============================================================================
# FINGER JOINT ANGLE CALCULATIONS (INDEX, MIDDLE, RING, PINKY)
# ============================================================================

def calculate_finger_mcp_flexion(
    landmarks: Sequence[Tuple[float, float, float]],
    mcp_idx: int,
    pip_idx: int
) -> float:
    """
    Calculate MCP joint flexion for a finger.

    Uses wrist→MCP as reference and MCP→PIP as the rotating segment.
    Projects onto the plane perpendicular to the knuckle line.

    Returns: angle in radians [0, π/2]
    """
    wrist = landmarks[LandmarkIndex.WRIST]
    mcp = landmarks[mcp_idx]
    pip = landmarks[pip_idx]

    # Define rotation axis (along the knuckle line)
    index_mcp = landmarks[LandmarkIndex.INDEX_MCP]
    pinky_mcp = landmarks[LandmarkIndex.PINKY_MCP]
    rotation_axis = vector_normalize(vector_subtract(pinky_mcp, index_mcp))

    # Reference vector (wrist to MCP)
    ref_vector = vector_subtract(mcp, wrist)

    # Finger segment (MCP to PIP)
    finger_vector = vector_subtract(pip, mcp)

    # Project into plane perpendicular to knuckle line
    ref_proj = project_vector_onto_plane(ref_vector, rotation_axis)
    finger_proj = project_vector_onto_plane(finger_vector, rotation_axis)

    # Normalize projected vectors
    ref_norm = vector_normalize(ref_proj)
    finger_norm = vector_normalize(finger_proj)

    # Unsigned angle between vectors
    dot = max(-1.0, min(1.0, vector_dot(ref_norm, finger_norm)))
    angle = math.acos(dot)

    return angle


def calculate_finger_pip_flexion(
    landmarks: Sequence[Tuple[float, float, float]],
    mcp_idx: int,
    pip_idx: int,
    dip_idx: int
) -> float:
    """
    Calculate PIP joint flexion for a finger.

    Measures angle between MCP→PIP and PIP→DIP segments.

    Returns: angle in radians [0, π]
    """
    mcp = landmarks[mcp_idx]
    pip = landmarks[pip_idx]
    dip = landmarks[dip_idx]

    # Define rotation axis (perpendicular to finger plane)
    index_mcp = landmarks[LandmarkIndex.INDEX_MCP]
    pinky_mcp = landmarks[LandmarkIndex.PINKY_MCP]
    rotation_axis = vector_normalize(vector_subtract(pinky_mcp, index_mcp))

    # Proximal phalanx (MCP to PIP)
    proximal = vector_subtract(pip, mcp)

    # Middle phalanx (PIP to DIP)
    middle = vector_subtract(dip, pip)

    # Calculate bending angle
    angle = signed_angle_in_plane(proximal, middle, rotation_axis)

    return math.pi - max(0.0, min(math.pi, angle))


def calculate_finger_dip_flexion(
    landmarks: Sequence[Tuple[float, float, float]],
    pip_idx: int,
    dip_idx: int,
    tip_idx: int
) -> float:
    """
    Calculate DIP joint flexion for a finger.

    Measures angle between PIP→DIP and DIP→TIP segments.

    Returns: angle in radians [0, π]
    """
    pip = landmarks[pip_idx]
    dip = landmarks[dip_idx]
    tip = landmarks[tip_idx]

    # Define rotation axis
    index_mcp = landmarks[LandmarkIndex.INDEX_MCP]
    pinky_mcp = landmarks[LandmarkIndex.PINKY_MCP]
    rotation_axis = vector_normalize(vector_subtract(pinky_mcp, index_mcp))

    # Middle phalanx (PIP to DIP)
    middle = vector_subtract(dip, pip)

    # Distal phalanx (DIP to TIP)
    distal = vector_subtract(tip, dip)

    # Calculate bending angle
    angle = signed_angle_in_plane(middle, distal, rotation_axis)

    return math.pi - max(0.0, min(math.pi, angle))


def calculate_all_joint_angles(landmarks: Sequence[Tuple[float, float, float]]) -> Dict[str, float]:
    """
    Calculate all joint angles from MediaPipe landmarks.

    Returns:
        Dictionary mapping joint names to angles in radians
    """
    angles = {}

    # Thumb joints (each calculated separately with vector projection)
    angles['thumb_cmc_abd'] = calculate_thumb_cmc_abduction(landmarks)
    angles['thumb_cmc_flex'] = calculate_thumb_cmc_flexion(landmarks)
    angles['thumb_mcp'] = calculate_thumb_mcp_flexion(landmarks)
    angles['thumb_ip'] = calculate_thumb_ip_flexion(landmarks)

    # Index finger
    angles['index_mcp'] = calculate_finger_mcp_flexion(
        landmarks, LandmarkIndex.INDEX_MCP, LandmarkIndex.INDEX_PIP
    )
    angles['index_pip'] = calculate_finger_pip_flexion(
        landmarks, LandmarkIndex.INDEX_MCP, LandmarkIndex.INDEX_PIP, LandmarkIndex.INDEX_DIP
    )
    angles['index_dip'] = calculate_finger_dip_flexion(
        landmarks, LandmarkIndex.INDEX_PIP, LandmarkIndex.INDEX_DIP, LandmarkIndex.INDEX_TIP
    )

    # Middle finger
    angles['middle_mcp'] = calculate_finger_mcp_flexion(
        landmarks, LandmarkIndex.MIDDLE_MCP, LandmarkIndex.MIDDLE_PIP
    )
    angles['middle_pip'] = calculate_finger_pip_flexion(
        landmarks, LandmarkIndex.MIDDLE_MCP, LandmarkIndex.MIDDLE_PIP, LandmarkIndex.MIDDLE_DIP
    )
    angles['middle_dip'] = calculate_finger_dip_flexion(
        landmarks, LandmarkIndex.MIDDLE_PIP, LandmarkIndex.MIDDLE_DIP, LandmarkIndex.MIDDLE_TIP
    )

    # Ring finger
    angles['ring_mcp'] = calculate_finger_mcp_flexion(
        landmarks, LandmarkIndex.RING_MCP, LandmarkIndex.RING_PIP
    )
    angles['ring_pip'] = calculate_finger_pip_flexion(
        landmarks, LandmarkIndex.RING_MCP, LandmarkIndex.RING_PIP, LandmarkIndex.RING_DIP
    )
    angles['ring_dip'] = calculate_finger_dip_flexion(
        landmarks, LandmarkIndex.RING_PIP, LandmarkIndex.RING_DIP, LandmarkIndex.RING_TIP
    )

    # Pinky finger
    angles['pinky_mcp'] = calculate_finger_mcp_flexion(
        landmarks, LandmarkIndex.PINKY_MCP, LandmarkIndex.PINKY_PIP
    )
    angles['pinky_pip'] = calculate_finger_pip_flexion(
        landmarks, LandmarkIndex.PINKY_MCP, LandmarkIndex.PINKY_PIP, LandmarkIndex.PINKY_DIP
    )
    angles['pinky_dip'] = calculate_finger_dip_flexion(
        landmarks, LandmarkIndex.PINKY_PIP, LandmarkIndex.PINKY_DIP, LandmarkIndex.PINKY_TIP
    )

    return angles

def angles_to_l20_pose(joint_angles: Dict[str, float]) -> List[int]:
    """
    Convert joint angles (radians) to L20 20-joint pose format.

    L20 Format (20 values, 0-255):
    [0-4]:   Base pitch (thumb, index, middle, ring, pinky) - main flexion
    [5-9]:   Abduction/adduction
    [10-14]: Thumb yaw + reserved
    [15-19]: Fingertip flexion

    Args:
        joint_angles: Dictionary with keys like 'thumb_mcp', 'index_pip', etc.

    Returns:
        List of 20 integer values (0-255)
    """
    pose = [255] * 20  # Start with fully open position

    # Helper to convert angle to motor value (0-255)
    # 0 radians = 255 (open), max_angle radians = 0 (closed)
    def angle_to_motor(angle: float, max_angle: float = math.pi) -> int:
        normalized = max(0.0, min(1.0, angle / max_angle))
        return int((1.0 - normalized) * 255)

    # Thumb control
    # [0] = Thumb base pitch (combination of MCP + IP)
    thumb_flex = (joint_angles.get('thumb_mcp', 0.0) + joint_angles.get('thumb_ip', 0.0)) / 2.0
    pose[0] = angle_to_motor(thumb_flex, math.pi)

    # [5] = Thumb abduction
    pose[5] = angle_to_motor(joint_angles.get('thumb_cmc_abd', 0.0), math.pi / 2)

    # [10] = Thumb yaw/rotation (from CMC flexion)
    pose[10] = angle_to_motor(joint_angles.get('thumb_cmc_flex', 0.0), math.pi / 2)

    # [15] = Thumb tip flexion
    pose[15] = angle_to_motor(joint_angles.get('thumb_ip', 0.0), math.pi)

    # Index finger
    index_flex = (joint_angles.get('index_mcp', 0.0) +
                  joint_angles.get('index_pip', 0.0) +
                  joint_angles.get('index_dip', 0.0)) / 3.0
    pose[1] = angle_to_motor(index_flex, math.pi)
    pose[16] = angle_to_motor(joint_angles.get('index_dip', 0.0), math.pi)

    # Middle finger
    middle_flex = (joint_angles.get('middle_mcp', 0.0) +
                   joint_angles.get('middle_pip', 0.0) +
                   joint_angles.get('middle_dip', 0.0)) / 3.0
    pose[2] = angle_to_motor(middle_flex, math.pi)
    pose[17] = angle_to_motor(joint_angles.get('middle_dip', 0.0), math.pi)

    # Ring finger
    ring_flex = (joint_angles.get('ring_mcp', 0.0) +
                 joint_angles.get('ring_pip', 0.0) +
                 joint_angles.get('ring_dip', 0.0)) / 3.0
    pose[3] = angle_to_motor(ring_flex, math.pi)
    pose[18] = angle_to_motor(joint_angles.get('ring_dip', 0.0), math.pi)

    # Pinky finger
    pinky_flex = (joint_angles.get('pinky_mcp', 0.0) +
                  joint_angles.get('pinky_pip', 0.0) +
                  joint_angles.get('pinky_dip', 0.0)) / 3.0
    pose[4] = angle_to_motor(pinky_flex, math.pi)
    pose[19] = angle_to_motor(joint_angles.get('pinky_dip', 0.0), math.pi)

    # Finger abduction (spread) - keep at neutral
    pose[6] = 100   # Index
    pose[7] = 180   # Middle
    pose[8] = 240   # Ring
    pose[9] = 255   # Pinky (slightly spread)

    # Reserved slots
    pose[11] = 255
    pose[12] = 255
    pose[13] = 255
    pose[14] = 255

    return pose
