import math
import numpy as np

from isaacgym import gymapi
from argparse import ArgumentParser

from locomotion.utils.utils import DTYPE
from locomotion.utils.isaacgym_wrapper import *

from locomotion.parameters import Parameters
from locomotion.estimator import gamepad_reader
from locomotion.controller.locomotion_controller import LocomotionController


parser = ArgumentParser(prog="LOCOMOTION_MAIN")
parser.add_argument("--robot", default="Aliengo", choices=["Aliengo", "A1", "GO1"], help="robot type")
parser.add_argument("--num-envs", type=int, default=1, help="number of robot environments")
parser.add_argument("--render-fps", type=int, default=30, help="render fps")

args = parser.parse_args()

debug_vis = False  # Draw ground normal vector

# Setup gamepad/keyboard input
gamepad = gamepad_reader.Gamepad(vel_scale_x=2.5, vel_scale_y=1.5, vel_scale_rot=3.0)
print("Keyboard input enabled")

def main():
    robot = RobotType[args.robot.upper()]
    dt = Parameters.controller_dt

    # Initialize Isaac Gym
    gym = gymapi.acquire_gym()
    sim = acquire_sim(gym, dt)

    # Add ground and terrain
    add_ground(gym, sim)
    add_terrain(gym, sim, "slope")
    add_terrain(gym, sim, "stair", 3.95, True)

    # Set up the simulation environment
    num_envs = args.num_envs
    envs_per_row = int(math.sqrt(args.num_envs))
    env_spacing = 0.5

    # Create environments
    envs, actors = create_envs(gym, sim, robot, num_envs, envs_per_row, env_spacing)
    
    # Setup camera
    cam_pos = gymapi.Vec3(2, 2, 2)
    viewer = add_viewer(gym, sim, envs[0], cam_pos)
    
    # Keyboard input setup
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "KEY_W")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "KEY_S")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "KEY_A")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "KEY_D")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "KEY_Q")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_E, "KEY_E")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "KEY_SPACE")
    try:
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ENTER, "KEY_ENTER")
    except AttributeError:
        gym.subscribe_viewer_keyboard_event(viewer, 13, "KEY_ENTER")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "KEY_ESCAPE")
    
    # Setup locomotion controllers
    controllers = []
    for idx in range(num_envs):
        # Configure joints for effort control mode
        props = gym.get_actor_dof_properties(envs[idx], actors[idx])
        props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
        props["stiffness"].fill(0.0)
        props["damping"].fill(0.0)
        gym.set_actor_dof_properties(envs[idx], actors[idx], props)
        
        # Initialize locomotion controller
        controller = LocomotionController()
        controller.init(robot)
        controllers.append(controller)
    
    # Simulation loop variables
    count = 0
    render_fps = args.render_fps
    render_count = int(1 / render_fps / Parameters.controller_dt)
    
    # Simulation loop
    while not gym.query_viewer_has_closed(viewer):
        # Handle keyboard events
        events = gym.query_viewer_action_events(viewer)
        for evt in events:
            if evt.value > 0:
                gamepad.handle_keyboard_action(evt.action, 'pressed')
            elif evt.value == 0:
                gamepad.handle_keyboard_action(evt.action, 'released')
        gamepad.update_keyboard_velocity()
        
        # Step physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        
        # Get velocity commands from gamepad
        commands = np.zeros(3, dtype=DTYPE)
        lin_speed, ang_speed, e_stop = gamepad.get_command()

        if not e_stop:
            commands = np.array([lin_speed[0], lin_speed[1], ang_speed], dtype=DTYPE)
        
        # Run controllers for each environment
        for idx, (env, actor, controller) in enumerate(zip(envs, actors, controllers)):
            # Get joint states
            dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
            
            # Get body states
            body_idx = gym.find_actor_rigid_body_index(env, actor, controller._quadruped._bodyName, gymapi.DOMAIN_ACTOR)
            body_states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)[body_idx]
            
            # Get leg torques
            leg_torques = controller.run(dof_states, body_states, commands).astype(np.float32)
            gym.apply_actor_dof_efforts(env, actor, leg_torques)
        
        # Debug visualization
        if debug_vis:
            pos_np = np.asarray([p for p in body_states["pose"]["p"]], dtype=np.float32)
            gym.add_lines(viewer, envs[0], 1, 
                [pos_np, pos_np + controllers[0]._state_estimator.result.ground_normal_world], 
                [[255,0,0]])
        
        # Render
        if count % render_count == 0:
            count = 0
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.clear_lines(viewer)
        
        # Sync frame time
        gym.sync_frame_time(sim)
        count += 1
    
    # Cleanup
    gamepad.stop()
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()

