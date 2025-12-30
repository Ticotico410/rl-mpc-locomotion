from locomotion.parameters import Parameters
from locomotion.controller.data_parser import DataParser
from locomotion.robot.quadruped import Quadruped, RobotType

from locomotion.planner.convexMPC import ConvexMPC
from locomotion.controller.PD_controller import LegController

from locomotion.estimator.state_estimator import StateEstimator
from locomotion.estimator.desired_state_command import DesiredStateCommand


class LocomotionController:
    def init(self, robot_type: RobotType):
        self._robot_type = robot_type

        if self._robot_type in RobotType:
            self._quadruped = Quadruped(self._robot_type)
        else:
            raise Exception("Invalid RobotType")

        # Initialize state estimator
        self._state_estimator = StateEstimator(self._quadruped)
        
        # Initialize leg controller
        self._leg_controller = LegController(self._quadruped)
        
        # Initialize desired state command
        self._desired_state_command = DesiredStateCommand()

        # Initialize controller data
        self.data = DataParser()
        self.data._quadruped = self._quadruped
        self.data._stateEstimator = self._state_estimator
        self.data._legController = self._leg_controller
        self.data._desiredStateCommand = self._desired_state_command

        # Initialize MPC planner
        self.mpc = ConvexMPC(
            dt=Parameters.controller_dt,
            iterationsBetweenMPC=Parameters.iterations_between_mpc/(1000.0*Parameters.controller_dt)
        )
        self.mpc.initialize(self.data)

    def reset(self):
        self.mpc.initialize(self.data)
        self._desired_state_command.reset()
        self._state_estimator.reset()
    
    def run(self, dof_states, body_states, commands):
        """Run one control iteration."""
        # Update desired commands
        self._desired_state_command.updateCommand(commands)

        # Update the joint states
        self._leg_controller.updateData(dof_states)
        self._leg_controller.zeroCommand()

        # Update robot states
        self._state_estimator.update(body_states)

        # Run the MPC planner
        self.mpc.run(self.data)

        # Set the leg controller commands
        leg_torques = self._leg_controller.updateCommand()

        return leg_torques
