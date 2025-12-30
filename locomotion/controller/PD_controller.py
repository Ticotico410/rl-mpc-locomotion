import numpy as np
from locomotion.utils.utils import DTYPE, getSideSign
from locomotion.robot.quadruped import Quadruped
from locomotion.robot.kinematics import LegKinematics


class LegControlData:
    def __init__(self):
        self.q = np.zeros((3,1), dtype=DTYPE)
        self.qd = np.zeros((3,1), dtype=DTYPE)

        self.J = np.zeros((3,3), dtype=DTYPE)
        self.p = np.zeros((3,1), dtype=DTYPE)

        self.v = np.zeros((3,1), dtype=DTYPE)

    def zero(self):
        self.q.fill(0)
        self.qd.fill(0)
        self.J.fill(0)
        self.p.fill(0)
        self.v.fill(0)
    
    def setQuadruped(self, quad:Quadruped):
        self.quadruped = quad


class LegControlCommand:
    def __init__(self):
        self.tau_ff = np.zeros((3,1), dtype=DTYPE)
        self.force_ff = np.zeros((3,1), dtype=DTYPE)

        # joint space
        self.q_ref = np.zeros((3,1), dtype=DTYPE)
        self.dq_ref = np.zeros((3,1), dtype=DTYPE)

        # cartesian space
        self.p_ref = np.zeros((3,1), dtype=DTYPE)
        self.v_ref = np.zeros((3,1), dtype=DTYPE) 
            
        # joint impedance control
        self.kp_joint = np.zeros((3,3), dtype=DTYPE)
        self.kd_joint = np.zeros((3,3), dtype=DTYPE)

        # cartesian impedance control
        self.kp_cartesian = np.zeros((3,3), dtype=DTYPE)
        self.kd_cartesian = np.zeros((3,3), dtype=DTYPE)

    def zero(self):
        self.tau_ff.fill(0)
        self.force_ff.fill(0)

        self.q_ref.fill(0)
        self.dq_ref.fill(0)
        self.p_ref.fill(0)
        self.v_ref.fill(0)

        self.kp_joint.fill(0)
        self.kd_joint.fill(0)
        self.kp_cartesian.fill(0)
        self.kd_cartesian.fill(0)


class LegController:
    def __init__(self, quad:Quadruped):
        self.commands = [LegControlCommand() for _ in range(4)]
        self.datas = [LegControlData() for _ in range(4)]

        self.kinematics = LegKinematics(quad._abadLinkLength, quad._hipLinkLength, quad._kneeLinkLength)

        self.max_torque = 0.0

        self._quadruped = quad
        for data in self.datas:
            data.setQuadruped(self._quadruped)

    def zeroCommand(self):
        "Initialize all command to zero at the beginning of each control loop"
        for command in self.commands:
            command.zero()

    def setMaxTorque(self, tau:float):
        self.max_torque = tau

    def updateData(self, dof_states):
        "Update leg data from simulator"
        positions = dof_states["pos"]  
        velocities = dof_states["vel"]  
        
        for leg in range(4):
            # Extract joint positions and velocities for this leg (3 joints per leg)
            self.datas[leg].q[:, 0] = positions[3 * leg : 3 * (leg + 1)]
            self.datas[leg].qd[:, 0] = velocities[3 * leg : 3 * (leg + 1)]

            # J and p (with side sign: +1 for left legs, -1 for right legs)
            side_sign = getSideSign(leg)
            p, J = self.kinematics.computePositionandJacobian(self.datas[leg].q, side_sign)
            self.datas[leg].p = p
            self.datas[leg].J = J

            # foot velocity
            self.datas[leg].v = self.datas[leg].J @ self.datas[leg].qd

    def updateCommand(self):
        "Update leg commands for simulator"
        legTorques = np.zeros(12, dtype=DTYPE)
        
        for leg in range(4):
            footForce = self.commands[leg].force_ff\
                        + self.commands[leg].kp_cartesian @ (self.commands[leg].p_ref - self.datas[leg].p)\
                        + self.commands[leg].kd_cartesian @ (self.commands[leg].v_ref - self.datas[leg].v)

            legTorque = self.commands[leg].tau_ff + self.datas[leg].J.T @ footForce\
                        + self.commands[leg].kp_joint @ (self.commands[leg].q_ref - self.datas[leg].q)\
                        + self.commands[leg].kd_joint @ (self.commands[leg].dq_ref - self.datas[leg].qd)

            legTorques[leg * 3 : (leg + 1) * 3] = legTorque.flatten()

        return legTorques