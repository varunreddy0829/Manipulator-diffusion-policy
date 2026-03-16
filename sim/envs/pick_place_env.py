# =============================================================================
# Stage 1: Pick-and-Place Simulation Environment
# =============================================================================
# This script creates a tabletop manipulation environment in Isaac Sim with:
#   - A Franka Panda 7-DOF robot arm
#   - A table for the robot to work on
#   - A red target cube to pick up
#   - A green goal zone to place the cube
#
# This is Stage 1 of the pipeline:
# Stage 1: Environment setup   
# Stage 2: Data collection
# Stage 3: Diffusion policy training
# Stage 4: Policy evaluation
# =============================================================================

from isaacsim import SimulationApp

# Initialize Isaac Sim FIRST before any other imports
# headless=False means we want the GUI viewport visible
simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid, FixedCuboid
from isaacsim.robot.manipulators.examples.franka import Franka
from pxr import UsdLux, Sdf
import omni.usd


def create_pick_place_env():
    """
    Creates the simulation environment with robot, table, and objects.
    
    Returns:
        world:  The Isaac Sim World object (manages physics + scene)
        robot:  The Franka robot object (lets us read/write joint states)
        target: The red cube object (lets us track its position)
    """

    # World is the main container for everything in the simulation
    # stage_units_in_meters=1.0 means 1 unit in USD = 1 meter in real life
    world = World(stage_units_in_meters=1.0)

    # Add a flat ground plane so objects don't fall into the void
    world.scene.add_ground_plane(size=4, color=np.array([0.5, 0.5, 0.5]))

    # Add lighting
    stage = omni.usd.get_context().get_stage()

    # Dome light - illuminates entire scene evenly
    dome_light = stage.DefinePrim("/World/DomeLight", "DomeLight")
    dome_light.GetAttribute("inputs:intensity").Set(1000.0)

    # -------------------------------------------------------------------------
    # Table: a flat brown visual box
    # FixedCuboid = has collision physics but doesn't move under gravity. Perfect for a table.
    # position Z=0.4 means the table CENTER is 40cm above the ground
    # scale [0.8, 0.8, 0.05] = 80cm wide, 80cm deep, 5cm thick
    # -------------------------------------------------------------------------
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Table",   # unique path in the USD stage
            name="table",               # name for Python reference
            position=np.array([0.0, 0.0, 0.4]),
            scale=np.array([1.5, 1.5, 0.05]),
            color=np.array([0.6, 0.4, 0.2]),  # brown
        )
    )

    # -------------------------------------------------------------------------
    # Franka Panda robot
    # Isaac Sim has a built-in Franka class with full articulation configured
    # position Z=0.425 = table top surface (table center 0.4 + half thickness 0.025)
    # The robot has 9 DOF: 7 arm joints + 2 finger joints
    # -------------------------------------------------------------------------
    robot = world.scene.add(
        Franka(
            prim_path="/World/franka",
            name="franka",
            position=np.array([0.0, 0.0, 0.425]),
        )
    )

    # -------------------------------------------------------------------------
    # Target cube: the object the robot will learn to pick up
    # DynamicCuboid = has physics (gravity, collisions, can be pushed/grabbed)
    # position Z=0.46 = just above the table surface so it rests on top
    # scale [0.04, 0.04, 0.04] = 4cm cube (realistic manipulation size)
    # -------------------------------------------------------------------------
    target = world.scene.add(
        DynamicCuboid(
            prim_path="/World/TargetCube",
            name="target_cube",
            position=np.array([0.4, 0.0, 0.448]),   # 40cm in front of robot
            scale=np.array([0.04, 0.04, 0.04]),
            color=np.array([1.0, 0.2, 0.2]),        # red
        )
    )

    # -------------------------------------------------------------------------
    # Goal zone: flat green marker showing WHERE to place the cube
    # VisualCuboid = no physics, just a visual target indicator
    # position Y=0.3 = 30cm to the side of the cube's start position
    # -------------------------------------------------------------------------
    world.scene.add(
        VisualCuboid(
            prim_path="/World/GoalZone",
            name="goal_zone",
            position=np.array([0.4, 0.3, 0.425]),
            scale=np.array([0.06, 0.06, 0.005]),    # flat thin marker
            color=np.array([0.2, 1.0, 0.2]),        # green
        )
    )

    return world, robot, target


if __name__ == "__main__":
    # Build the environment
    world, robot, target = create_pick_place_env()

    # Reset initializes all physics, joints, and object positions
    # Must be called before stepping the simulation
    world.reset()

    # Confirm the robot loaded correctly
    # Franka should have 9 DOF: 7 arm + 2 fingers
    print(f"Robot DOF count: {robot.num_dof}")
    print(f"Joint names: {robot.dof_names}")
    print("Environment ready!")

    # Run the simulation indefinitely until the window is closed
    # world.step(render=True) advances physics by one timestep and renders
    while simulation_app.is_running():
        world.step(render=True)

    # Clean shutdown
    simulation_app.close()