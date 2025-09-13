# 🚀 RobotController – Task-Space Path Tracking with CLIK + Dubins Paths

This repository implements a **robot controller** for articulated robots (using [Robotics Toolbox for Python](https://github.com/petercorke/robotics-toolbox-python)) that can follow smooth Cartesian paths in the **x–z plane**.  
It combines:

- **Task-space Control**: Closed-Loop Inverse Kinematics (CLIK) at the **acceleration level** with PD gains.
- **Dubins Path Planning**: Generates feasible paths through waypoints, ignoring headings if desired.
- **Damped Least Squares (DLS)** pseudoinverse to handle Jacobian singularities robustly.
- **Dynamics**: Uses full robot dynamics (`rne`) for torque commands (inertia, Coriolis, gravity).

The controller can track long waypoint paths (e.g., 1000+ samples) and includes optimizations for **speed** and **robustness**.

---

## ✨ Features
- Dubins-based waypoint connection (with optional "home" start).
- Task-space PD control with acceleration-level CLIK.
- DLS-based Jacobian pseudoinverse with adaptive damping.
- Full robot inverse dynamics for torque computation.
- Torque saturation for safety.
- Debug plotting of generated paths and poses.
- Configurable speed-up options:
  - Path downsampling
  - Windowed index skipping
  - Curvature-aware compression
  - Cubic spline resampling for smooth derivatives

---

## 📂 Code Overview
- `RobotController`  
  Main class handling:
  - **Path initialization** (`initialize_new_path`)  
  - **Control loop** (`control_step`)  
  - **Path generation** with Dubins planner  
  - **Kinematic and dynamic helpers**  

Key methods:
- `dubins_through_poses` → Build Dubins paths through waypoints.
- `get_direct_kinematics`, `get_cartesian_vel`, `get_jdot_qdot` → Task-space kinematics.
- `get_jacobian_pseudo_inverse` → Robust pseudoinverse with DLS.
- `control_step` → Returns torque command for the current state.
- `plot_poses` → Visualize path and orientation.

---

## ⚙️ Usage
```python
import roboticstoolbox as rtb
import numpy as np
from robot_controller import RobotController  # your file

# Example: 2-link planar arm
robot = rtb.models.DH.Planar2()

controller = RobotController(robot, dt=0.01)

# Define waypoints in x–z plane
waypoints = np.array([
    [0.3, 0.2],
    [0.6, 0.5],
    [0.9, 0.3]
])

# Initialize new path
controller.initialize_new_path(waypoints)

# Simulate
q = np.zeros(robot.n)
dq = np.zeros(robot.n)
for t in range(1000):
    tau = controller.control_step(q, dq)
    # integrate robot dynamics or pass tau to simulator
