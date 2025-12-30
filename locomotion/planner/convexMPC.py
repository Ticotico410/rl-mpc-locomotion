import sys
import time
from typing import Any
import numpy as np

from locomotion.parameters import Parameters
from locomotion.planner.gait import OffsetDurationGait
from locomotion.planner.foot_swing_trajectory import FootSwingTrajectory
from locomotion.controller.data_parser import DataParser

from locomotion.utils.utils import DTYPE, NUM_LEGS, getSideSign, CASTING
from locomotion.utils.orientation_tools import coordinateRotation, CoordinateAxis

try:
    import mpc_osqp as mpc
except:
    print("Run 'pip install -e .' to install the OSQP solver dependencies.")
    sys.exit()


class ConvexMPC:
    def __init__(self, dt: float, iterationsBetweenMPC: int):
        self.iterationsBetweenMPC = int(iterationsBetweenMPC)
        self.horizonLength = Parameters.mpc_horizon_length
        self.dt = dt
        
        # Initialize gait
        self.trotting = OffsetDurationGait(10, 
                            np.array([0, 5, 5, 0], dtype=DTYPE), 
                            np.array([5, 5, 5, 5], dtype=DTYPE), "Trotting")
        
        self.dtMPC = self.dt * self.iterationsBetweenMPC
        self.default_iterations_between_mpc = self.iterationsBetweenMPC
        print("[Convex MPC] dt: %.3f iterations: %d, dtMPC: %.3f" % (self.dt, self.iterationsBetweenMPC, self.dtMPC))
        
        self.firstSwing:list = None
        self.firstRun = True
        self.iterationCounter = 0
        self.pFoot = np.zeros((4, 3, 1), dtype=DTYPE)
        self.f_ff = np.zeros((4, 3, 1), dtype=DTYPE)

        self.foot_positions = np.zeros((4, 3, 1), dtype=DTYPE)

        self._x_vel_des = 0.0
        self._y_vel_des = 0.0
        self._yaw_turn_rate = 0.0

        self._roll_des = 0.0
        self._pitch_des = 0.0

        self.footSwingTrajectories = [FootSwingTrajectory() for _ in range(4)]
        self.swingTimes = np.zeros((4,1), dtype=DTYPE)
        self.swingTimeRemaining = [0.0 for _ in range(4)]

        self.Kp_swing = Parameters.kp_swing_cartesian
        self.Kd_swing = Parameters.kd_swing_cartesian
        self.Kp_stance = Parameters.kp_stance_cartesian
        self.Kd_stance = Parameters.kd_stance_cartesian

        # MPC solver (will be initialized in initialize method)
        self._cpp_mpc = None
 
    def initialize(self, data:DataParser):
        """Initialize MPC solver with robot parameters."""
        if Parameters.mpc_alpha > 1e-4:
            print("Alpha was set too high (" + str(Parameters.mpc_alpha) + ") adjust to 1e-5\n")
            Parameters.mpc_alpha = 1e-5

        self.iterationCounter = 0
        self._cpp_mpc = mpc.ConvexMpc(
            data._quadruped._bodyMass,
            list(data._quadruped._bodyInertia),
            NUM_LEGS,
            self.horizonLength,
            self.dtMPC,
            Parameters.mpc_alpha,
            mpc.QPOASES
        )

        self._x_vel_des = 0.0
        self._y_vel_des = 0.0
        self._yaw_turn_rate = 0.0
        self.firstSwing = [True for _ in range(4)]
        self.firstRun = True

    def recomputerTiming(self, iterations_per_mpc:int):
        self.iterationsBetweenMPC = iterations_per_mpc
        self.dtMPC = self.dt*iterations_per_mpc

    def __SetupCommand(self, data:DataParser):
        """Update desired velocities from command"""
        self._body_height = data._quadruped._bodyHeight
        self._x_vel_des = data._desiredStateCommand.x_vel_cmd
        self._y_vel_des = data._desiredStateCommand.y_vel_cmd
        self._yaw_turn_rate = data._desiredStateCommand.z_rot_cmd

    def solveDenseMPC(self, mpcTable:list, data:DataParser):
        """Solve MPC optimization problem."""
        seResult = data._stateEstimator.getResult()
        
        # *MPC Weights
        if data._desiredStateCommand.mpc_weights is None:
            mpc_weight = Parameters.mpc_weights
        else:
            mpc_weight = data._quadruped._mpc_weights

        timer = time.time()

        # *Normal Vector of ground (assume flat ground)
        if Parameters.flat_ground:
            gravity_projection_vec = np.array([0, 0, 1], dtype=DTYPE)
        else:
            gravity_projection_vec = seResult.ground_normal_yaw
        
        # *Google's way of states
        com_roll_pitch_yaw = seResult.rpy_body.flatten()
        com_position = seResult.pos.flatten()
        com_angular_velocity = seResult.omega_body.flatten()
        com_velocity = seResult.v_body.flatten()

        desired_com_position = np.array([0., 0., self._body_height], dtype=DTYPE)
        desired_com_velocity = np.array([self._x_vel_des, self._y_vel_des, 0], dtype=DTYPE)
        desired_com_roll_pitch_yaw = np.zeros(3, dtype=DTYPE) # walk parallel to the ground
        desired_com_angular_velocity = np.array([0, 0, self._yaw_turn_rate], dtype=DTYPE)

        predicted_contact_forces = self._cpp_mpc.compute_contact_forces(
            mpc_weight,                                            # MPC input weights
            com_position,                                          # position of COM
            com_velocity,                                          # velocity of COM
            com_roll_pitch_yaw,                                    # RPY of the COM
            gravity_projection_vec,                                # normal vector of ground
            com_angular_velocity,                                  # angular velocity of COM
            np.asarray(mpcTable, dtype=DTYPE),                     # foot contact states
            np.array(self.foot_positions.flatten(), dtype=DTYPE),  # foot positions
            data._quadruped._friction_coeffs,                      # foot friction coefficients
            desired_com_position,                                  # desired position of COM
            desired_com_velocity,                                  # desired velocity of COM
            desired_com_roll_pitch_yaw,                            # desired RPY of COM
            desired_com_angular_velocity                           # desired angular velocity of COM
            )
        for leg in range(4):
            self.f_ff[leg] = np.array(predicted_contact_forces[leg * 3 : (leg + 1) * 3], dtype=DTYPE).reshape((3,1))

    def updateMPCIfNeeded(self, mpcTable:list, data:DataParser):
        """Update MPC solution if needed (based on iteration counter)"""
        if(self.iterationCounter % self.iterationsBetweenMPC) == 0:
            self.solveDenseMPC(mpcTable, data)

    def run(self, data:DataParser):
        # Command Setup
        self.__SetupCommand(data)
        gaitNumber = Parameters.cmpc_gait.value
        seResult = data._stateEstimator.getResult()

        # pick gait
        gait = self.trotting
        self.current_gait = gaitNumber

        # set gait iterations
        gait.setIterations(self.iterationsBetweenMPC, self.iterationCounter)
        self.recomputerTiming(self.default_iterations_between_mpc)

        for i in range(4):
            self.foot_positions[i] = data._quadruped.getHipLocation(i) + data._legController.datas[i].p
            self.pFoot[i] = self.foot_positions[i] + seResult.pos

        # * first time initialization
        if self.firstRun:
            self.firstRun = False
            data._stateEstimator._init_contact_history(self.foot_positions)
            for i in range(4):
                self.footSwingTrajectories[i].setHeight(0.05)
                self.footSwingTrajectories[i].setInitialPosition(self.pFoot[i])
                self.footSwingTrajectories[i].setFinalPosition(self.pFoot[i])

        if Parameters.flat_ground:
            data._stateEstimator._update_com_position_ground_frame(self.foot_positions)
        else:
            data._stateEstimator._compute_ground_normal_and_com_position(self.foot_positions)
        
        # * foot placement
        for l in range(4):
            self.swingTimes[l] = gait.getCurrentSwingTime(self.dtMPC, l)

        v_des_robot = np.array([self._x_vel_des, self._y_vel_des, 0], dtype=DTYPE).reshape((3,1))

        """Raibert Heuristic method for foot placement"""
        for i in range(4):
            if self.firstSwing[i]:
                self.swingTimeRemaining[i] = self.swingTimes[i].item()
            else:
                self.swingTimeRemaining[i] -= self.dt

            self.footSwingTrajectories[i].setHeight(self._body_height/3)
            
            offset = np.array([0, getSideSign(i)*data._quadruped._abadLinkLength, 0], dtype=DTYPE).reshape((3,1))
            pRobotFrame = data._quadruped.getHipLocation(i) + offset
            stance_time = gait.getCurrentStanceTime(self.dtMPC, i)
            
            pYawCorrected = coordinateRotation(CoordinateAxis.Z, -self._yaw_turn_rate*stance_time/2) @ pRobotFrame
            Pf = seResult.pos + (pYawCorrected + v_des_robot * self.swingTimeRemaining[i])

            p_rel_max = 0.3
            pfx_rel = seResult.v_body[0, 0] * (0.5 + Parameters.cmpc_bonus_swing) * stance_time + \
                      0.03 * (seResult.v_body[0, 0] - v_des_robot[0, 0]) + \
                      (0.5 * seResult.pos[2, 0] / -9.81) * (seResult.v_body[1, 0] * self._yaw_turn_rate)
            
            pfy_rel = seResult.v_body[1, 0] * 0.5 * stance_time * self.dtMPC + \
                      0.03 * (seResult.v_body[1, 0] - v_des_robot[1, 0]) + \
                      (0.5 * seResult.pos[2, 0] / -9.81) * (-seResult.v_body[0, 0] * self._yaw_turn_rate)
            
            pfx_rel = min(max(pfx_rel, -p_rel_max), p_rel_max)
            pfy_rel = min(max(pfy_rel, -p_rel_max), p_rel_max)
            Pf[0] += pfx_rel
            Pf[1] += pfy_rel
            Pf[2] = -0.003
            self.footSwingTrajectories[i].setFinalPosition(Pf)

        # calc gait
        self.iterationCounter += 1

        # gait
        contactStates = gait.getContactState()
        swingStates = gait.getSwingState()
        mpcTable = gait.getMpcTable()

        # * update MPC
        self.updateMPCIfNeeded(mpcTable, data)

        se_contactState = np.array([0,0,0,0], dtype=DTYPE).reshape((4,1))

        for foot in range(4):
            contactState = contactStates[foot]
            swingState = swingStates[foot]

            # Swing leg control
            if swingState > 0:
                if self.firstSwing[foot]:
                    self.firstSwing[foot] = False
                    self.footSwingTrajectories[foot].setInitialPosition(self.pFoot[foot])

                self.footSwingTrajectories[foot].computeSwingTrajectoryBezier(swingState, self.swingTimes[foot].item())
                pDesFoot = self.footSwingTrajectories[foot].getPosition()
                vDesFoot = self.footSwingTrajectories[foot].getVelocity()

                pDesLeg = (pDesFoot - seResult.pos) - data._quadruped.getHipLocation(foot)
                vDesLeg = (vDesFoot - seResult.v_body)

                np.copyto(data._legController.commands[foot].p_ref, pDesLeg, casting=CASTING)
                np.copyto(data._legController.commands[foot].v_ref, vDesLeg, casting=CASTING)
                np.copyto(data._legController.commands[foot].kp_cartesian, self.Kp_swing, casting=CASTING)
                np.copyto(data._legController.commands[foot].kd_cartesian, self.Kd_swing, casting=CASTING)

            # Stance leg control
            else:
                self.firstSwing[foot] = True
                pDesFoot = self.footSwingTrajectories[foot].getPosition()
                vDesFoot = self.footSwingTrajectories[foot].getVelocity()

                pDesLeg = (pDesFoot - seResult.pos) - data._quadruped.getHipLocation(foot)
                vDesLeg = (vDesFoot - seResult.v_body)
                
                np.copyto(data._legController.commands[foot].p_ref, pDesLeg, casting=CASTING)
                np.copyto(data._legController.commands[foot].v_ref, vDesLeg, casting=CASTING)
                np.copyto(data._legController.commands[foot].kp_cartesian, self.Kp_stance, casting=CASTING)
                np.copyto(data._legController.commands[foot].kd_cartesian, self.Kd_stance, casting=CASTING)
                np.copyto(data._legController.commands[foot].force_ff, self.f_ff[foot], casting=CASTING)
                np.copyto(data._legController.commands[foot].kd_joint, np.eye(3, dtype=DTYPE)*0.2, casting=CASTING)

                se_contactState[foot] = contactState

        data._stateEstimator.setContactPhase(se_contactState)
