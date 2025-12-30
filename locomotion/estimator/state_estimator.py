import numpy as np
from scipy import linalg

from locomotion.parameters import Parameters
from locomotion.robot.quadruped import Quadruped

from locomotion.utils.utils import DTYPE, NUM_LEGS
from locomotion.utils.orientation_tools import *


class StateEstimate:
    def __init__(self):
        self.pos = np.zeros((3, 1), dtype=DTYPE)  # Position
        self.ori = Quaternion(1, 0, 0, 0)         # Orientation
        
        # World frame velocities
        self.v_world = np.zeros((3, 1), dtype=DTYPE)
        self.omega_world = np.zeros((3, 1), dtype=DTYPE)
        
        # Body frame velocities
        self.v_body = np.zeros((3, 1), dtype=DTYPE)
        self.omega_body = np.zeros((3, 1), dtype=DTYPE)
        
        # Rotation matrices and angles
        self.body_R_world = np.zeros((3, 3), dtype=DTYPE)    # Rotation matrix from world frame to body frame
        self.rpy = np.zeros((3, 1), dtype=DTYPE)       # RPY in world frame
        self.rpy_body = np.zeros((3, 1), dtype=DTYPE)  # RPY in yaw-aligned ground frame
        
        # Ground normal vectors
        self.ground_normal_yaw = np.array([0, 0, 1], dtype=DTYPE)    # In yaw-aligned frame
        self.ground_normal_world = np.array([0, 0, 1], dtype=DTYPE)  # In world frame


class StateEstimator:
    def __init__(self, quad: Quadruped):
        self._quad = quad
        self.result = StateEstimate()
        
        # Internal state
        self._contact_phase = np.zeros((NUM_LEGS, 1), dtype=DTYPE)
        self._foot_contact_history = None
        self.ground_R_body = None  # Rotation from body to ground frame
        
        # Body height
        self.body_height = self._quad._bodyHeight
        self.result.pos[2] = self.body_height

    def reset(self):
        """Reset state estimator"""
        self.result = StateEstimate()
        self._phase = np.zeros((NUM_LEGS, 1), dtype=DTYPE)
        self._contact_phase = np.zeros((NUM_LEGS, 1), dtype=DTYPE)
        self._foot_contact_history = None
        self.ground_R_body = None
        self.body_height = self._quad._bodyHeight
        self.result.pos[2] = self.body_height

    def setContactPhase(self, phase: np.ndarray):
        """Set contact phase for each leg (1 = contact, 0 = swing)"""
        self._contact_phase = phase

    def getResult(self):
        """Get current state estimate result"""
        return self.result

    def update(self, body_states):
        """Update state estimate from body states."""
        # Update velocities in world frame
        for idx in range(3):
            self.result.v_world[idx] = body_states["vel"]["linear"][idx]
            self.result.omega_world[idx] = body_states["vel"]["angular"][idx]

        # Update orientation (Quat: x, y, z, w)
        self.result.ori.x = body_states["pose"]["r"][0]
        self.result.ori.y = body_states["pose"]["r"][1]
        self.result.ori.z = body_states["pose"]["r"][2]
        self.result.ori.w = body_states["pose"]["r"][3]

        # Compute rotation matrix from world frame to body frame
        self.result.body_R_world = quat_to_rot(self.result.ori)
        
        # Transform velocities from world frame to body frame
        self.result.v_body = self.result.body_R_world @ self.result.v_world
        self.result.omega_body = self.result.body_R_world @ self.result.omega_world

        # Compute RPY in world frame
        self.result.rpy = quat_to_rpy(self.result.ori)
       
        # Build rotation from world frame to yaw-aligned frame
        yaw_R_world = rpy_to_rot([0, 0, self.result.rpy[2]])
        
        # Build rotation from yaw-aligned frame to ground frame
        ground_R_yaw = get_rot_from_normals(np.array([0, 0, 1], dtype=DTYPE), self.result.ground_normal_yaw)
        
        # Build rotation from body frame to ground frame
        self.ground_R_body = self.result.body_R_world @ yaw_R_world.T @ ground_R_yaw.T

        # Compute RPY of body in ground frame
        self.result.rpy_body = rot_to_rpy(self.ground_R_body)

    def _init_contact_history(self, foot_positions: np.ndarray):
        """Initialize foot contact history."""
        foot_pos = np.asarray(foot_positions).reshape(NUM_LEGS, 3)
        self._foot_contact_history = foot_pos.copy()
        self._foot_contact_history[:, 2] = -self.body_height

    def _update_contact_history(self, foot_positions: np.ndarray):
        """Update foot contact history with current contact positions."""
        foot_pos = np.asarray(foot_positions).reshape(NUM_LEGS, 3)
        for leg_id in range(NUM_LEGS):
            if self._contact_phase[leg_id]:
                self._foot_contact_history[leg_id] = foot_pos[leg_id]

    def _update_com_position_ground_frame(self, foot_positions: np.ndarray):
        """Update CoM position in ground frame based on foot contacts."""
        foot_contacts = self._contact_phase.flatten()
        if np.sum(foot_contacts) == 0:
            # No contact, use default height
            return np.array((0, 0, self.body_height))
        else:
            # Transform foot positions from body frame to yaw-aligned ground frame
            foot_pos = np.asarray(foot_positions).reshape(NUM_LEGS, 3)
            foot_positions_ground_frame = foot_pos.dot(self.ground_R_body.T)
            foot_heights = -foot_positions_ground_frame[:, 2]
            
        # Average height of contact feet
        height_in_ground_frame = np.sum(foot_heights * foot_contacts) / np.sum(foot_contacts)
        self.result.pos[2] = height_in_ground_frame

    def _compute_ground_normal_and_com_position(self, foot_positions:np.ndarray):
        """
        Computes the surface orientation in robot frame based on foot positions.
        Solves a least squares problem, see the following paper for details:
        https://ieeexplore.ieee.org/document/7354099
        """
        self._update_com_position_ground_frame(foot_positions)
        self._update_contact_history(foot_positions)

        contact_foot_positions = self._foot_contact_history.reshape((4,3)) # reshape from (4,3,1) to (4,3)
        normal_vec = linalg.lstsq(contact_foot_positions, np.ones(4, dtype=DTYPE))[0]
        normal_vec /= np.linalg.norm(normal_vec)
        if normal_vec[2] < 0:
            normal_vec = -normal_vec

        _ground_normal = normal_vec
        _ground_normal /= np.linalg.norm(_ground_normal)

        # ground normal in yaw aligned ground frame and world frame
        self.result.ground_normal_yaw = _ground_normal
        self.result.ground_normal_world = self.result.body_R_world.T @ _ground_normal
