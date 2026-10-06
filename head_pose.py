"""Head rotation from MediaPipe's facial transformation matrix.

The matrix maps the canonical face into camera space, where +X is image
right, +Y is up and +Z points from the scene towards the camera.  Angles
are reported from the performer's own point of view (un-mirrored):

    yaw   > 0   head turned to their left
    pitch > 0   looking up
    roll  > 0   head tipped towards their left shoulder

so that R = Ry(yaw) * Rx(-pitch) * Rz(-roll).  The avatar uses the same
convention for its own left/right; mirroring is applied later.
"""
import math


def matrix_to_angles(matrix):
    """(yaw, pitch, roll) in radians from a 4x4 (or 3x3) transform."""
    cols = []
    for c in range(3):
        col = [float(matrix[r][c]) for r in range(3)]
        length = math.sqrt(sum(v * v for v in col)) or 1.0
        cols.append([v / length for v in col])      # drop any scale
    r = [[cols[c][row] for c in range(3)] for row in range(3)]
    pitch = math.asin(min(max(r[1][2], -1.0), 1.0))
    yaw = math.atan2(r[0][2], r[2][2])
    roll = math.atan2(-r[1][0], r[1][1])
    return yaw, pitch, roll


def mirror_angles(angles):
    """Reflection left-right: yaw and roll change sign, pitch does not."""
    yaw, pitch, roll = angles
    return -yaw, pitch, -roll
