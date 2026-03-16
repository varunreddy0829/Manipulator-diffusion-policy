# Stage 2: Data Collection — Problems & Solutions

## Overview
This document captures every significant problem encountered during Stage 2
(keyboard teleoperation + data collection in Isaac Sim 4.5) and the solution
that resolved each one. Useful for debugging, onboarding, and portfolio documentation.

---

## Problem 1 — `get_keyboard()` attribute error

**Error:**
```
AttributeError: 'carb.input.IInput' object has no attribute 'get_keyboard'
```

**Cause:**
The `carb.input` API changed in Isaac Sim 4.5. The old way of acquiring
the keyboard device directly from `carb.input.IInput` no longer works.

**Solution:**
Use `omni.appwindow` to get the keyboard instead:
```python
# Wrong (old API)
self.input    = carb.input.acquire_input_interface()
self.keyboard = self.input.get_keyboard()

# Correct (Isaac Sim 4.5)
self._appwindow = omni.appwindow.get_default_app_window()
self._input     = carb.input.acquire_input_interface()
self._keyboard  = self._appwindow.get_keyboard()
```

---

## Problem 2 — Camera segfault on initialization

**Error:**
```
Fatal Python error: Segmentation fault
./python.sh: line 41: Segmentation fault (core dumped)
```

**Cause:**
The `isaacsim.sensors.camera` Camera class was being initialized before
the simulation had fully settled, causing a memory access violation.

**Solution:**
Remove camera initialization for Stage 2 and use a placeholder zero image.
Camera will be added properly in a later iteration after the core
teleoperation pipeline is stable:
```python
# Placeholder instead of camera
img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
```

---

## Problem 3 — `ArticulationAction` import not found

**Error:**
```
ImportError: cannot import name 'ArticulationAction' from
'isaacsim.core.api.controllers'
```

**Cause:**
In Isaac Sim 4.5, `ArticulationAction` moved to a different module path
compared to older versions.

**Solution:**
```python
# Wrong
from isaacsim.core.api.controllers import ArticulationAction

# Correct
from isaacsim.core.utils.types import ArticulationAction
```

---

## Problem 4 — Robot spawns in weird/collapsed configuration

**Symptom:**
Franka arm spawns bent awkwardly or going through the table instead of
standing in a natural upright home position.

**Cause:**
RMPflow initializes from whatever joint configuration the robot is in
at startup. Without explicitly setting home joints, physics initializes
the robot in a default zero-angle configuration which looks collapsed.

**Solution:**
Set home joint positions explicitly and step the simulation 30 times
to let physics settle BEFORE initializing RMPflow:
```python
home_joints = np.array([
    0.0, -0.785, 0.0, -2.356,
    0.0,  1.571,  0.785,
    GRIPPER_OPEN, GRIPPER_OPEN
])
robot.set_joint_positions(home_joints)
for _ in range(30):
    world.step(render=False)

# Then initialize RMPflow AFTER settling
rmp_flow = RmpFlow(...)
```

Also get the actual end effector position after settling as the
starting target — don't hardcode it:
```python
ee_pos, ee_ori   = robot.end_effector.get_world_pose()
ee_target_pos    = ee_pos.copy()
ee_target_ori    = ee_ori.copy()
```

---

## Problem 5 — Direct joint control only moves base joint

**Symptom:**
Pressing arrow keys only moved joint 1 and joint 3. The arm didn't
move naturally in Cartesian space.

**Cause:**
Mapping keyboard deltas directly to individual joint angles is not
intuitive — each joint controls rotation in a different plane and
the mapping changes depending on the current configuration.

**Solution:**
Use RMPflow (Riemannian Motion Policy) for inverse kinematics.
Move the end effector target in Cartesian space and let RMPflow
solve all joint angles automatically:
```python
from isaacsim.robot_motion.motion_generation import (
    RmpFlow, ArticulationMotionPolicy
)

rmp_flow = RmpFlow(
    robot_description_path=ROBOT_DESC,
    rmpflow_config_path=RMP_CONFIG,
    urdf_path=URDF_PATH,
    end_effector_frame_name="right_gripper",
    maximum_substep_size=0.00167,
)
articulation_rmp = ArticulationMotionPolicy(
    robot, rmp_flow, default_physics_dt=1/60.0
)

# In the loop — move EE target then apply IK
ee_target_pos += delta
rmp_flow.set_end_effector_target(ee_target_pos, ee_target_ori)
action = articulation_rmp.get_next_articulation_action()
robot.apply_action(action)
```

RMPflow config files location in Isaac Sim 4.5:
```
~/isaac-sim/exts/isaacsim.robot_motion.motion_generation/
  motion_policy_configs/franka/rmpflow/
    franka_rmpflow_common.yaml
    robot_descriptor.yaml
```

---

## Problem 6 — Robot arm swinging/oscillating during movement

**Symptom:**
While moving the arm with arrow keys, the arm occasionally swings or
oscillates slightly, especially during continuous key holds.

**Cause:**
RMPflow is a reactive policy — it computes forces at every timestep
toward the moving target. When the target moves faster than the arm
can follow, it overshoots and oscillates. Also occurs near singular
joint configurations.

**Solution:**
Two fixes applied together:

1. Reduce `maximum_substep_size` for smoother integration:
```python
maximum_substep_size=0.00167  # halved from default 0.00334
```

2. Smooth the target position with exponential moving average:
```python
# Instead of direct addition
ee_target_pos = ee_target_pos + delta

# Use smoothed update
ee_target_pos = 0.85 * ee_target_pos + 0.15 * (ee_target_pos + delta)
```

---

## Problem 7 — Recording too many idle steps (2000 steps per episode)

**Symptom:**
Each episode recorded ~2000 steps even for a simple pick and place,
because data was recorded every timestep including all idle time
(camera rotation, thinking, pausing).

**Cause:**
The original recording logic used `step % 3 != 0` which records
every 3rd simulation step regardless of whether the robot was moving.

**Solution:**
Only record when the robot is actually moving — check if any
keyboard delta is non-zero or if the gripper state changed:
```python
gripper_changed = (gripper_cmd != prev_gripper)
robot_moving    = np.any(np.abs(delta) > 0)
prev_gripper    = gripper_cmd

if not (robot_moving or gripper_changed):
    # Don't record — just handle save/reset/quit flags
    save, reset, quit_ = controller.consume_flags()
else:
    # Record this step
    dataset.add_step(image=img, state=state, action=action)
    save, reset, quit_ = controller.consume_flags()
```

Result: Episodes reduced from ~2000 steps to ~100-700 meaningful steps.

---

## Problem 8 — Key tap required for every movement step

**Symptom:**
Holding down the S key (Z down) only moved the arm one step then
stopped. Had to tap the key repeatedly for continuous movement —
very tedious for collecting demos.

**Cause:**
The original `_on_key` handler set `self.delta` on KEY_PRESS and
reset it to zero on KEY_RELEASE. Since OS key repeat fires
KEY_PRESS/KEY_RELEASE pairs rapidly, movement was jerky and required
constant tapping.

**Solution:**
Track a set of currently held keys and compute delta from that set
every simulation step. This gives truly continuous smooth movement
while any key is held:
```python
def _on_key(self, event, *args):
    pressed  = (event.type == carb.input.KeyboardEventType.KEY_PRESS)
    released = (event.type == carb.input.KeyboardEventType.KEY_RELEASE)
    k = event.input

    if pressed:
        self._held_keys.add(k)
    if released:
        self._held_keys.discard(k)

def get_delta(self):
    delta = np.zeros(3)
    if carb.input.KeyboardInput.UP    in self._held_keys:
        delta[0] += STEP_SIZE_XY
    if carb.input.KeyboardInput.DOWN  in self._held_keys:
        delta[0] -= STEP_SIZE_XY
    if carb.input.KeyboardInput.W     in self._held_keys:
        delta[2] += STEP_SIZE_Z
    if carb.input.KeyboardInput.S     in self._held_keys:
        delta[2] -= STEP_SIZE_Z
    # ... etc
    return delta, self.gripper_cmd
```

---

## Problem 9 — Z movement too slow

**Symptom:**
Moving the arm down to reach the cube required 150+ keypresses even
with `STEP_SIZE_Z = 0.020`. Had to travel ~250mm vertically which
took a very long time.

**Root cause:**
This was actually caused by Problem 8 — since keys weren't held
continuously, each "hold" only registered as one step. After fixing
the held-keys issue, the arm moves at `STEP_SIZE_Z` per simulation
step continuously, which is much faster.

**Final step size config:**
```python
STEP_SIZE_XY = 0.015   # meters per sim step for X/Y
STEP_SIZE_Z  = 0.012   # meters per sim step for Z
```

---

## Final working config summary

```python
# Step sizes
STEP_SIZE_XY = 0.015
STEP_SIZE_Z  = 0.012

# Workspace clamp (safe operating range for Franka on table)
EE_MIN = [0.2, -0.4, 0.43]
EE_MAX = [0.7,  0.4,  0.9]

# RMPflow
maximum_substep_size = 0.00167

# Recording
record_only_when_moving = True

# Target demos
TARGET_DEMOS = 50
```

---

## Key lessons learned

1. Always set robot home position and step physics before initializing
   motion controllers — they depend on a valid starting configuration.

2. Use IK (RMPflow) for teleoperation, never direct joint control —
   Cartesian space is intuitive, joint space is not.

3. Only record meaningful data — idle steps pollute the dataset and
   teach the policy to do nothing.

4. Track held keys with a set, not key press/release deltas — this
   is the standard pattern for smooth real-time keyboard control.

5. Test imports in headless mode before writing full scripts — saves
   debugging time significantly.
