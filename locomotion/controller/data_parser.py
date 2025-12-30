from locomotion.robot.quadruped import Quadruped
from locomotion.controller.PD_controller import LegController
from locomotion.estimator.state_estimator import StateEstimator
from locomotion.estimator.desired_state_command import DesiredStateCommand

class DataParser:
    def __init__(self):
        self._quadruped:Quadruped = None
        self._stateEstimator:StateEstimator = None
        self._legController:LegController = None
        self._desiredStateCommand:DesiredStateCommand = None

