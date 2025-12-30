class DesiredStateCommand:
    def __init__(self):
        self.x_vel_cmd = 0.0
        self.y_vel_cmd = 0.0
        self.z_rot_cmd = 0.0
        self.mpc_weights = None
    
    def updateCommand(self, commands):
        self.x_vel_cmd = commands[0]
        self.y_vel_cmd = commands[1]
        self.z_rot_cmd = commands[2]
    
    def reset(self):
        self.x_vel_cmd = 0.0
        self.y_vel_cmd = 0.0
        self.z_rot_cmd = 0.0