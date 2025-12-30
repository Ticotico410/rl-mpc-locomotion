import time


class Gamepad:
  """Interface for reading commands from keyboard via Isaac Gym.

  Keyboard control:
  - W/S: forward/backward
  - A/D: left/right
  - Q/E: rotate left/right
  - ESC: emergency stop
  """
  def __init__(self, 
               vel_scale_x: float=.5, 
               vel_scale_y: float=.5, 
               vel_scale_rot: float=1.):
    """Initialize the keyboard controller.
    Args:
      vel_scale_x: maximum absolute x-velocity command.
      vel_scale_y: maximum absolute y-velocity command.
      vel_scale_rot: maximum absolute yaw-dot command.
    """
    print("Keyboard control instructions:")
    print("  W/S: forward/backward")
    print("  A/D: left/right")
    print("  Q/E: rotate left/right")
    print("  ESC: stop")
    
    self._vel_scale_x = float(vel_scale_x)
    self._vel_scale_y = float(vel_scale_y)
    self._vel_scale_rot = float(vel_scale_rot)

    # Controller states
    self.vx, self.vy, self.wz = 0., 0., 0.
    self._estop_flagged = False
    self.is_running = True
    
    # Keyboard state tracking
    self._key_states = {
      'w': False, 's': False, 'a': False, 'd': False,
      'q': False, 'e': False
    }

  def handle_keyboard_input(self, key):
    """Handle keyboard input for special keys (ESC)."""
    if key == '\x1b':  # ESC key
      print("EStop Flagged, press Enter to release.")
      self._estop_flagged = True
      self.vx, self.vy, self.wz = 0., 0., 0.
      for k in self._key_states:
        self._key_states[k] = False
      return
    elif key == '\n' or key == '\r':  # Enter
      if self._estop_flagged:
        print("Estop Released.")
        self._estop_flagged = False
      return

  def update_keyboard_velocity(self):
    """Update velocity commands based on current key states."""
    if self._estop_flagged:
      self.vx, self.vy, self.wz = 0., 0., 0.
      for k in self._key_states:
        self._key_states[k] = False
      return
    
    self.vx = 0.0
    self.vy = 0.0
    self.wz = 0.0
    
    if self._key_states['w']:
      self.vx = self._vel_scale_x
    elif self._key_states['s']:
      self.vx = -self._vel_scale_x
    
    if self._key_states['a']:
      self.vy = self._vel_scale_y
    elif self._key_states['d']:
      self.vy = -self._vel_scale_y
    
    if self._key_states['q']:
      self.wz = self._vel_scale_rot
    elif self._key_states['e']:
      self.wz = -self._vel_scale_rot

  def handle_keyboard_action(self, action_name, action_type):
    """Handle keyboard action event from Isaac Gym viewer.
    Args:
      action_name: Action name string (e.g., "KEY_W", "KEY_S")
      action_type: 'pressed' or 'released'
    """
    action_map = {
      "KEY_W": 'w',
      "KEY_S": 's',
      "KEY_A": 'a',
      "KEY_D": 'd',
      "KEY_Q": 'q',
      "KEY_E": 'e',
      "KEY_ENTER": '\n',
      "KEY_ESCAPE": '\x1b',
    }
    
    if action_name in action_map:
      key_char = action_map[action_name]
      
      if action_type == 'pressed':
        if key_char in ['\n', '\x1b']:
          self.handle_keyboard_input(key_char)
        elif key_char in self._key_states:
          self._key_states[key_char] = True
          self.update_keyboard_velocity()
      elif action_type == 'released':
        if key_char in self._key_states:
          self._key_states[key_char] = False
          self.update_keyboard_velocity()

  def get_command(self):
    return (self.vx, self.vy, 0), self.wz, self._estop_flagged

  def stop(self):
    self.is_running = False
