import itertools
import time
from MPC_Controller.Parameters import Parameters
from MPC_Controller.utils import GaitType, FSM_StateName


ALLOWED_MODES = [FSM_StateName.RECOVERY_STAND, FSM_StateName.LOCOMOTION]
ALLOWED_GAITS = [x for x in GaitType]


class Gamepad:
  """Interface for reading commands from keyboard via Isaac Gym.

  Keyboard control:
  - W/S: forward/backward
  - A/D: left/right
  - Q/E: rotate left/right
  - Space: switch gait
  - Enter: switch mode
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
    print("键盘控制说明:")
    print("  W/S: 前进/后退")
    print("  A/D: 左移/右移")
    print("  Q/E: 左转/右转")
    print("  空格: 切换步态")
    print("  回车: 切换模式")
    print("  ESC: 紧急停止")
    
    self._vel_scale_x = float(vel_scale_x)
    self._vel_scale_y = float(vel_scale_y)
    self._vel_scale_rot = float(vel_scale_rot)

    self._gait_generator = itertools.cycle(ALLOWED_GAITS)
    self._gait = next(self._gait_generator)
    self._mode_generator = itertools.cycle(ALLOWED_MODES)
    self._mode = next(self._mode_generator)

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
    """Handle keyboard input for special keys (Space, Enter, ESC)."""
    # 处理特殊按键
    if key == '\x1b':  # ESC key
      print("EStop Flagged, press Enter to release.")
      self._estop_flagged = True
      self.vx, self.vy, self.wz = 0., 0., 0.
      self._mode = FSM_StateName.RECOVERY_STAND
      # 清除所有按键状态
      for k in self._key_states:
        self._key_states[k] = False
      return
    elif key == ' ':  # Space
      if not self._estop_flagged:
        self._gait = next(self._gait_generator)
        print(f"切换步态: {self._gait}")
      return
    elif key == '\n' or key == '\r':  # Enter
      if self._estop_flagged:
        print("Estop Released.")
        self._estop_flagged = False
        self._mode_generator = itertools.cycle(ALLOWED_MODES)
        self._mode = next(self._mode_generator)
      else:
        self._mode = next(self._mode_generator)
        print(f"切换模式: {self._mode}")
      return

  def update_keyboard_velocity(self):
    """Update velocity commands based on current key states."""
    if self._estop_flagged:
      self.vx, self.vy, self.wz = 0., 0., 0.
      # 清除所有按键状态
      for k in self._key_states:
        self._key_states[k] = False
      return
    
    # 根据按键状态更新速度
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
    # 映射action名称到按键字符
    action_map = {
      "KEY_W": 'w',
      "KEY_S": 's',
      "KEY_A": 'a',
      "KEY_D": 'd',
      "KEY_Q": 'q',
      "KEY_E": 'e',
      "KEY_SPACE": ' ',
      "KEY_ENTER": '\n',
      "KEY_ESCAPE": '\x1b',
    }
    
    if action_name in action_map:
      key_char = action_map[action_name]
      
      if action_type == 'pressed':
        if key_char in [' ', '\n', '\x1b']:
          # 特殊按键立即处理
          self.handle_keyboard_input(key_char)
        elif key_char in self._key_states:
          # 普通按键更新状态为按下
          self._key_states[key_char] = True
          # 立即更新速度
          self.update_keyboard_velocity()
      elif action_type == 'released':
        if key_char in self._key_states:
          # 普通按键更新状态为释放
          self._key_states[key_char] = False
          # 立即更新速度
          self.update_keyboard_velocity()

  def get_command(self):
    return (self.vx, self.vy, 0), self.wz, self._estop_flagged

  def get_gait(self):
    return self._gait

  def get_mode(self):
    return self._mode

  def stop(self):
    self.is_running = False
