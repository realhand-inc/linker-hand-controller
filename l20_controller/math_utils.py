from typing import Sequence, Tuple
import math

def vector_subtract(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    """Compute vector a - b."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vector_add(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    """Compute vector a + b."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vector_dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def vector_cross(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    """Compute cross product a × b."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vector_normalize(v: Sequence[float]) -> Tuple[float, float, float]:
    """Normalize vector to unit length."""
    magnitude = math.sqrt(sum(x * x for x in v))
    if magnitude < 1e-8:
        return (0.0, 0.0, 1.0)  # Default to z-axis if zero vector
    return (v[0] / magnitude, v[1] / magnitude, v[2] / magnitude)


def vector_magnitude(v: Sequence[float]) -> float:
    """Calculate magnitude of vector."""
    return math.sqrt(sum(x * x for x in v))


def project_vector_onto_plane(vector: Sequence[float], plane_normal: Sequence[float]) -> Tuple[float, float, float]:
    """
    Project a vector onto a plane defined by its normal vector.

    Formula: v_projected = v - (v · n)n
    where n is the unit normal to the plane
    """
    normal = vector_normalize(plane_normal)
    dot_product = vector_dot(vector, normal)
    projection_on_normal = (normal[0] * dot_product, normal[1] * dot_product, normal[2] * dot_product)
    return vector_subtract(vector, projection_on_normal)


def palm_plane_normal(
    middle_mcp: Sequence[float],
    wrist: Sequence[float],
    index_mcp: Sequence[float],
    pinky_mcp: Sequence[float],
) -> Tuple[float, float, float]:
    """
    Calculate a normalized palm plane normal using:
    cross(middle_mcp - wrist, index_mcp - pinky_mcp).
    """
    middle_minus_wrist = vector_subtract(middle_mcp, wrist)
    index_minus_pinky = vector_subtract(index_mcp, pinky_mcp)
    return vector_normalize(vector_cross(middle_minus_wrist, index_minus_pinky))


def signed_angle_in_plane(v1: Sequence[float], v2: Sequence[float], plane_normal: Sequence[float]) -> float:
    """
    Calculate signed angle between two vectors projected onto a plane.

    Returns angle in radians [-π, π].
    The sign is determined by the plane normal using right-hand rule.
    """
    # Project both vectors onto the plane
    v1_proj = project_vector_onto_plane(v1, plane_normal)
    v2_proj = project_vector_onto_plane(v2, plane_normal)

    # Normalize projected vectors
    v1_norm = vector_normalize(v1_proj)
    v2_norm = vector_normalize(v2_proj)

    # Calculate unsigned angle
    dot = max(-1.0, min(1.0, vector_dot(v1_norm, v2_norm)))
    angle = math.acos(dot)

    # Determine sign using cross product
    cross = vector_cross(v1_norm, v2_norm)
    sign = vector_dot(cross, plane_normal)

    return angle if sign >= 0 else -angle
