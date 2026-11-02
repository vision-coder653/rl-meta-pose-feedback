"""A right-handed finger frame from its distal direction and flexion axis."""
from __future__ import annotations

import numpy as np


def finger_frame(longitudinal_axis, flexion_axis) -> np.ndarray:
    """Columns are +X toward the fingertip, +Y along flexion, +Z = X cross Y.

    Project the longitudinal direction into the hinge's perpendicular plane.
    This removes dependence on where the anchor lies along the hinge axis.
    Both inputs must be fixed in the same rigid link frame; an extended finger
    does not make this construction singular.
    """
    longitudinal = np.asarray(longitudinal_axis, dtype=np.float64)
    flexion = np.asarray(flexion_axis, dtype=np.float64)
    if (longitudinal.shape != (3,) or flexion.shape != (3,)
            or not np.isfinite(longitudinal).all() or not np.isfinite(flexion).all()):
        raise ValueError("Finger axes must be finite three-dimensional vectors")
    length, axis_length = np.linalg.norm(longitudinal), np.linalg.norm(flexion)
    if length == 0 or axis_length == 0:
        raise ValueError("Finger axes must be nonzero")
    x = longitudinal / length
    y = flexion / axis_length
    x = x - y * np.dot(x, y)
    if np.linalg.norm(x) <= 64 * np.finfo(np.float64).eps:
        raise ValueError("Finger longitudinal and flexion axes must not be parallel")
    x /= np.linalg.norm(x)
    return np.column_stack((x, y, np.cross(x, y)))
