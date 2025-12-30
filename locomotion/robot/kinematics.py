import numpy as np
from math import sin, cos
from locomotion.utils.utils import DTYPE


class LegKinematics:
    def __init__(self, l1:float, l2:float, l3:float):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        
    def computePositionandJacobian(self, q:np.ndarray, side_sign:float = 1.0):
        """
        return position and Jacobian of the leg
        
        Args:
            q: joint angles (3,) array
            side_sign: +1 for left legs, -1 for right legs
        """
        dy = self.l1 * side_sign
        dz1 = -self.l2
        dz2 = -self.l3 
        
        s1 = sin(q[0])
        s2 = sin(q[1])
        s3 = sin(q[2])
        c1 = cos(q[0])
        c2 = cos(q[1])
        c3 = cos(q[2])

        c23 = c2 * c3 - s2 * s3
        s23 = s2 * c3 + c2 * s3

        # position
        p = np.zeros((3, 1), dtype=DTYPE)
        p[0] = dz2 * s23 + dz1 * s2
        p[1] = dy * c1 - dz1 * c2 * s1 - dz2 * s1 * c23
        p[2] = dy * s1 + dz1 * c1 * c2 + dz2 * c1 * c23

        # Jacobian
        J = np.zeros((3, 3), dtype=DTYPE)
        J[0, 0] = 0.0
        J[1, 0] = -dy * s1 - dz2 * c1 * c23 - dz1 * c1 * c2
        J[2, 0] = -dz2 * s1 * c23 + dy * c1 - dz1 * c2 * s1

        J[0, 1] = dz2 * c23 + dz1 * c2
        J[1, 1] = dz2 * s1 * s23 + dz1 * s1 * s2
        J[2, 1] = -dz2 * c1 * s23 - dz1 * c1 * s2

        J[0, 2] = dz2 * c23
        J[1, 2] = dz2 * s1 * s23
        J[2, 2] = -dz2 * c1 * s23

        return p, J
