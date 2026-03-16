# =============================================================================
# Stage 2: Keyboard Teleoperation + Data Collection (with RMPflow IK)
# =============================================================================
# Controls:
#   Arrow keys     → move end effector X/Y
#   W / S          → move end effector Z up/down
#   G              → toggle gripper open/close
#   R              → reset episode (discard current)
#   SPACE          → save episode and reset
#   Q              → quit
# =============================================================================

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import zarr
import os
import carb.input
import omni.appwindow
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid, FixedCuboid
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot_motion.motion_generation import RmpFlow, ArticulationMotionPolicy
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.extensions import get_extension_path_from_name

# =============================================================================
# CONFIG
# =============================================================================
TARGET_DEMOS   = 50
DATA_DIR       = os.path.expanduser(
    "~/RSE_Projects/Manipulator-diffusion-policy/data"
)
SAVE_PATH      = os.path.join(DATA_DIR, "demos.zarr")
IMG_SIZE       = 84
ACTION_DIM     = 9
STEP_SIZE      = 0.008   # meters per keypress — smaller = more precise
STEP_SIZE_XY = 0.015  # X/Y movement — medium speed
STEP_SIZE_Z  = 0.012   # Z movement — slower for precise grasping
GRIPPER_OPEN   = 0.04
GRIPPER_CLOSED = 0.0

# RMPflow config paths
MOTION_GEN_EXT = (
    "~/isaac-sim/exts/isaacsim.robot_motion.motion_generation"
)
RMP_CONFIG_DIR = os.path.expanduser(
    f"{MOTION_GEN_EXT}/motion_policy_configs/franka/rmpflow"
)
ROBOT_DESC     = os.path.join(RMP_CONFIG_DIR, "robot_descriptor.yaml")
RMP_CONFIG     = os.path.join(RMP_CONFIG_DIR, "franka_rmpflow_common.yaml")
URDF_PATH      = os.path.expanduser(
    "~/isaac-sim/exts/isaacsim.robot_motion.motion_generation"
    "/motion_policy_configs/franka/lula_franka_gen.urdf"
)

os.makedirs(DATA_DIR, exist_ok=True)

# =============================================================================
# SCENE SETUP
# =============================================================================
def build_scene():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_ground_plane(size=2.0, color=np.array([0.5, 0.5, 0.5]))

    stage = omni.usd.get_context().get_stage()
    dome  = stage.DefinePrim("/World/DomeLight", "DomeLight")
    dome.GetAttribute("inputs:intensity").Set(1000.0)

    world.scene.add(FixedCuboid(
        prim_path="/World/Table", name="table",
        position=np.array([0.0, 0.0, 0.4]),
        scale=np.array([1.5, 1.5, 0.05]),
        color=np.array([0.6, 0.4, 0.2]),
    ))

    robot = world.scene.add(Franka(
        prim_path="/World/franka", name="franka",
        position=np.array([0.0, 0.0, 0.425]),
    ))

    target = world.scene.add(DynamicCuboid(
        prim_path="/World/TargetCube", name="target_cube",
        position=np.array([0.4, 0.0, 0.448]),
        scale=np.array([0.04, 0.04, 0.04]),
        color=np.array([1.0, 0.2, 0.2]),
    ))

    world.scene.add(VisualCuboid(
        prim_path="/World/GoalZone", name="goal_zone",
        position=np.array([0.4, 0.3, 0.425]),
        scale=np.array([0.08, 0.08, 0.005]),
        color=np.array([0.2, 1.0, 0.2]),
    ))

    return world, robot, target


# =============================================================================
# KEYBOARD CONTROLLER
# =============================================================================
class KeyboardController:
    def __init__(self):
        self._appwindow  = omni.appwindow.get_default_app_window()
        self._input      = carb.input.acquire_input_interface()
        self._keyboard   = self._appwindow.get_keyboard()

        self._held_keys   = set()
        self.gripper_cmd  = GRIPPER_OPEN
        self.save_episode  = False
        self.reset_episode = False
        self.quit          = False

        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_key
        )

    def _on_key(self, event, *args):
        pressed  = (event.type == carb.input.KeyboardEventType.KEY_PRESS)
        released = (event.type == carb.input.KeyboardEventType.KEY_RELEASE)
        k = event.input

        if pressed:
            self._held_keys.add(k)
            # One-shot actions on press only
            if k == carb.input.KeyboardInput.G:
                self.gripper_cmd = (
                    GRIPPER_CLOSED
                    if self.gripper_cmd == GRIPPER_OPEN
                    else GRIPPER_OPEN
                )
            elif k == carb.input.KeyboardInput.SPACE:
                self.save_episode = True
            elif k == carb.input.KeyboardInput.R:
                self.reset_episode = True
            elif k == carb.input.KeyboardInput.Q:
                self.quit = True

        if released:
            self._held_keys.discard(k)

    def get_delta(self):
        """Compute delta from currently held keys — smooth continuous movement."""
        delta = np.zeros(3)

        if carb.input.KeyboardInput.UP    in self._held_keys:
            delta[0] += STEP_SIZE_XY
        if carb.input.KeyboardInput.DOWN  in self._held_keys:
            delta[0] -= STEP_SIZE_XY
        if carb.input.KeyboardInput.LEFT  in self._held_keys:
            delta[1] += STEP_SIZE_XY
        if carb.input.KeyboardInput.RIGHT in self._held_keys:
            delta[1] -= STEP_SIZE_XY
        if carb.input.KeyboardInput.W     in self._held_keys:
            delta[2] += STEP_SIZE_Z
        if carb.input.KeyboardInput.S     in self._held_keys:
            delta[2] -= STEP_SIZE_Z

        return delta, self.gripper_cmd

    def consume_flags(self):
        save  = self.save_episode
        reset = self.reset_episode
        quit_ = self.quit
        self.save_episode  = False
        self.reset_episode = False
        return save, reset, quit_


# =============================================================================
# ZARR DATASET
# =============================================================================
class DemoDataset:
    """
    Stores episodes in zarr format compatible with Lerobot trainer.

    Structure:
      demos.zarr/
        action          (N, 9)
        episode_idx     (N,)
        timestamp       (N,)
        observation/
          image         (N, 84, 84, 3)
          state         (N, 18)
    """
    def __init__(self, path):
        self.path           = path
        self.root           = zarr.open(path, mode='a')
        self.episode_buffer = []

        if "action" not in self.root:
            self.root.create_dataset(
                "action", shape=(0, ACTION_DIM),
                chunks=(1000, ACTION_DIM), dtype='f4'
            )
            obs = self.root.require_group("observation")
            obs.create_dataset(
                "image",
                shape=(0, IMG_SIZE, IMG_SIZE, 3),
                chunks=(100, IMG_SIZE, IMG_SIZE, 3),
                dtype='u1'
            )
            obs.create_dataset(
                "state", shape=(0, 18),
                chunks=(1000, 18), dtype='f4'
            )
            self.root.create_dataset(
                "episode_idx", shape=(0,),
                chunks=(1000,), dtype='i4'
            )
            self.root.create_dataset(
                "timestamp", shape=(0,),
                chunks=(1000,), dtype='i4'
            )

        if len(self.root["episode_idx"]) > 0:
            self.episode_count = int(self.root["episode_idx"][-1]) + 1
        else:
            self.episode_count = 0

        print(f"Dataset ready. Episodes so far: {self.episode_count}")

    def add_step(self, image, state, action):
        self.episode_buffer.append({
            "image":  image,
            "state":  state,
            "action": action,
        })

    def save_episode(self):
        if len(self.episode_buffer) < 5:
            print("Episode too short — discarding")
            self.episode_buffer = []
            return False

        n       = len(self.episode_buffer)
        images  = np.stack([s["image"]  for s in self.episode_buffer])
        states  = np.stack([s["state"]  for s in self.episode_buffer])
        actions = np.stack([s["action"] for s in self.episode_buffer])
        ep_idx  = np.full(n, self.episode_count, dtype=np.int32)
        ts      = np.arange(n, dtype=np.int32)

        self.root["action"].append(actions)
        self.root["observation/image"].append(images)
        self.root["observation/state"].append(states)
        self.root["episode_idx"].append(ep_idx)
        self.root["timestamp"].append(ts)

        self.episode_count += 1
        self.episode_buffer = []
        print(f"Saved episode {self.episode_count}/{TARGET_DEMOS} "
              f"({n} steps)")
        return True

    def discard_episode(self):
        n = len(self.episode_buffer)
        self.episode_buffer = []
        print(f"Discarded episode ({n} steps)")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    world, robot, target = build_scene()
    world.reset()

    # Set robot to home position and let physics settle
    home_joints = np.array([
        0.0, -0.785, 0.0, -2.356,
        0.0,  1.571,  0.785,
        GRIPPER_OPEN, GRIPPER_OPEN
    ])
    robot.set_joint_positions(home_joints)
    for _ in range(30):
        world.step(render=False)

    # ── Setup RMPflow IK controller ──
    rmp_flow = RmpFlow(
        robot_description_path=ROBOT_DESC,
        rmpflow_config_path=RMP_CONFIG,
        urdf_path=URDF_PATH,
        end_effector_frame_name="right_gripper",
        maximum_substep_size=0.00167,
    )
    articulation_rmp = ArticulationMotionPolicy(
        robot, rmp_flow, default_physics_dt=1 / 60.0
    )

    # Tell RMPflow where the robot base is in the world
    robot_base_translation, robot_base_orientation = robot.get_world_pose()
    rmp_flow.set_robot_base_pose(
        robot_base_translation, robot_base_orientation
    )

    # Get actual EE position after settling — use as starting target
    ee_pos, ee_ori = robot.end_effector.get_world_pose()
    ee_target_pos  = ee_pos.copy()
    ee_target_ori  = ee_ori.copy()

    # ── Keyboard and dataset ──
    controller  = KeyboardController()
    dataset     = DemoDataset(SAVE_PATH)
    prev_gripper = GRIPPER_OPEN

    print("\n" + "="*50)
    print("DATA COLLECTION READY (IK mode)")
    print(f"Goal: {TARGET_DEMOS} demos | Done: {dataset.episode_count}")
    print("-"*50)
    print("Arrow keys  → end effector X/Y")
    print("W / S       → end effector Z up/down")
    print("G           → toggle gripper")
    print("SPACE       → save episode")
    print("R           → discard + reset")
    print("Q           → quit")
    print("="*50 + "\n")

    step = 0

    while simulation_app.is_running():
        world.step(render=True)
        step += 1

        # ── Update end effector target from keyboard ──
        delta, gripper_cmd = controller.get_delta()
        # Smooth the target position with exponential moving average
        ee_target_pos = 0.85 * ee_target_pos + 0.15 * (ee_target_pos + delta)

        # Clamp to safe workspace
        ee_target_pos = np.clip(
            ee_target_pos,
            [0.2, -0.4, 0.43],
            [0.7,  0.4,  0.9]
        )

        # ── Apply IK via RMPflow ──
        rmp_flow.set_end_effector_target(
            target_position=ee_target_pos,
            target_orientation=ee_target_ori,
        )
        action_joints = articulation_rmp.get_next_articulation_action()
        robot.apply_action(action_joints)

        # ── Set gripper separately ──
        robot.gripper.apply_action(
            ArticulationAction(
                joint_positions=np.array([gripper_cmd, gripper_cmd])
            )
        )

        # ── Only record when robot is actually moving ──
        gripper_changed = (gripper_cmd != prev_gripper)
        robot_moving    = np.any(np.abs(delta) > 0)
        prev_gripper    = gripper_cmd

        if not (robot_moving or gripper_changed):
            # Still handle save/reset/quit even when not moving
            save, reset, quit_ = controller.consume_flags()
        else:
            # ── Robot state ──
            joint_pos = robot.get_joint_positions()
            joint_vel = robot.get_joint_velocities()
            state     = np.concatenate(
                [joint_pos, joint_vel]
            ).astype(np.float32)

            # ── Record action as current joint positions ──
            action = joint_pos.copy().astype(np.float32)

            # ── Image placeholder ──
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

            dataset.add_step(image=img, state=state, action=action)

            # Print step count every 10 recorded steps
            if len(dataset.episode_buffer) % 10 == 0 and \
               len(dataset.episode_buffer) > 0:
                print(f"  Steps recorded: {len(dataset.episode_buffer)}")

            save, reset, quit_ = controller.consume_flags()

        if save:
            saved = dataset.save_episode()
            if saved:
                x = 0.35 + np.random.uniform(-0.05, 0.05)
                y = np.random.uniform(-0.1, 0.1)
                target.set_world_pose(
                    position=np.array([x, y, 0.448])
                )
                ee_pos, ee_ori = robot.end_effector.get_world_pose()
                ee_target_pos  = ee_pos.copy()
                robot.set_joint_positions(home_joints)
                remaining = TARGET_DEMOS - dataset.episode_count
                print(f"Episodes remaining: {remaining}")

        if reset:
            dataset.discard_episode()
            robot.set_joint_positions(home_joints)
            ee_pos, ee_ori = robot.end_effector.get_world_pose()
            ee_target_pos  = ee_pos.copy()
            print("Reset — starting fresh episode")

        if quit_ or dataset.episode_count >= TARGET_DEMOS:
            print(f"\nDone! {dataset.episode_count} episodes → {SAVE_PATH}")
            break

    simulation_app.close()