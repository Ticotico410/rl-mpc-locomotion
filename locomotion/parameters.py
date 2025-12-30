import numpy as np
from locomotion.utils.utils import DTYPE, GaitType


class Parameters:
    """ System Parameters """
    controller_dt = 0.01         # control loop timestep
    iterations_between_mpc = 27  # period between MPC updates
    mpc_alpha = 1e-5             # regularization weight for contact forces

    """ Horizon Length """
    mpc_horizon_length = 10 
    
    """ MPC Weights """
    mpc_weights = np.array([
        0.25,  # roll
        0.25,  # pitch
        10,    # yaw

        2.0,   # x
        2.0,   # y
        50.0,  # z

        0.0,   # roll_dot
        0.0,   # pitch_dot
        0.3,   # yaw_dot

        0.5,   # vx
        0.5,   # vy
        0.1,   # vz
        
        0.0    # gravity
    ], dtype=DTYPE)
    
    """ PD Control Parameters """   
    # Swing phase PD gains (cartesian space)
    kp_swing_cartesian = np.array([
        700, 0, 0,
        0, 700, 0,
        0, 0, 150], dtype=DTYPE).reshape((3, 3))
    
    kd_swing_cartesian = np.array([
        7, 0, 0,
        0, 7, 0,
        0, 0, 7], dtype=DTYPE).reshape((3, 3))
    
    # Stance phase PD gains (cartesian space)
    kp_stance_cartesian = np.zeros((3, 3), dtype=DTYPE)
    kd_stance_cartesian = kd_swing_cartesian.copy()
    
    # Stance phase joint damping (joint space)   
    kd_joint_stance = 0.2
    
    """ Body Height """
    body_height = 0.35

    """ Ground Normal Vector """
    ground_normal = np.array([0.0, 0.0, 1.0], dtype=DTYPE)
    flat_ground = True

    """ Foot Swing Parameters """
    swing_height = 0.05             # Maximum swing height for foot trajectory
    swing_height_ratio = 1.0 / 3.0  # Ratio of body height for swing height (if used)
    
    foot_placement_max_offset = 0.3       # Maximum relative foot placement offset
    foot_placement_velocity_gain = 0.03   # Gain for velocity-based foot placement adjustment
    foot_placement_cross_term_gain = 0.5  # Gain for cross-coupling terms in foot placement
    foot_ground_clearance = -0.003        # Small ground clearance for foot placement
    
    # Convex MPC bonus swing
    cmpc_bonus_swing = 0.0
    
    """ Gait Parameters """
    trot_offset = np.array([0, 5, 5, 0], dtype=DTYPE)    # [FL, FR, BL, BR] offset
    trot_duration = np.array([5, 5, 5, 5], dtype=DTYPE)  # [FL, FR, BL, BR] stance duration
    

    """ MPC Optimization Parameters """
    cmpc_py_solver = 1
    cmpc_solver_type = 2
    cmpc_gait = GaitType.TROT