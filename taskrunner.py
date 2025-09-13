import os
import numpy as np
from robotcontroller_template import RobotController
import pybullet as pb
import roboticstoolbox as rtb
import time

def create_robot(urdf_file):
    robot_id = pb.loadURDF(urdf_file, useFixedBase=True)
    urdf_file = os.path.abspath(urdf_file)
    twolink = rtb.Robot.URDF(urdf_file)
    return robot_id, twolink

def run_task(task_file, robot_id, robot_controller, dt, fast_eval=False):
    # Load task data (desired EE path)
    data = np.loadtxt(task_file)
    print(f"Running task on data from {task_file}")

    # Reset robot to initial position
    pb.resetJointState(robot_id, 1, 0)
    pb.resetJointState(robot_id, 2, 0)

    # Create list for storing poses
    poses_list = []

    # Initialize the path in the controller
    robot_controller.initialize_new_path(data)
    # Run the control loop
    for step in range(1_000_000):
        if not robot_controller.is_running():
            break
        # Get current joint states
        state_joint1 = pb.getJointState(robot_id, 1)
        state_joint2 = pb.getJointState(robot_id, 2)
        q_current = np.array([state_joint1[0], state_joint2[0]])
        
        poses_list.append(q_current)
        # Get current joint velocities
        dq_current = np.array([state_joint1[1], state_joint2[1]])

        # Compute control output with your(!) controller
        control_output = robot_controller.control_step(q_current, dq_current)
        
        # limit torque
        control_output = np.clip(control_output, -50.0, 50.0)

        # Apply control output to the robot in pybullet
        pb.setJointMotorControl2(robot_id, 1, controlMode=pb.TORQUE_CONTROL, force=control_output[0])
        pb.setJointMotorControl2(robot_id, 2, controlMode=pb.TORQUE_CONTROL, force=control_output[1])
        pb.stepSimulation()
        if not fast_eval:
            time.sleep(dt/10)
    return poses_list


def main():
    # Check if tasks dir is present
    if not os.path.isdir("tasks"):
        print("Tasks dir is not present")
    os.makedirs("task_out", exist_ok=True)
    fast_eval = os.getenv("FAST_EVAL") is not None
    tasks = [os.path.join("tasks", f) for f in os.listdir("tasks")]
    pb.connect(pb.GUI if not fast_eval else pb.DIRECT)
    pb.resetSimulation()
    pb.setGravity(0, 0, -9.81)

    # Create Two Link Planar Robot
    robot_id, robot = create_robot("robot.urdf")

    # Set up robot parameters
    pb.setRealTimeSimulation(0)
    pb.enableJointForceTorqueSensor(robot_id, 1)
    pb.enableJointForceTorqueSensor(robot_id, 2)
    dt = 0.01
    pb.setTimeStep(dt)
    pb.setJointMotorControl2(robot_id, 1, controlMode=pb.VELOCITY_CONTROL, force=0.)
    pb.setJointMotorControl2(robot_id, 2, controlMode=pb.VELOCITY_CONTROL, force=0.)
    pb.setJointMotorControl2(robot_id, 1, controlMode=pb.TORQUE_CONTROL, force=0.)
    pb.setJointMotorControl2(robot_id, 2, controlMode=pb.TORQUE_CONTROL, force=0.)

    # Create the robot controller with the robot model and the time step
    robot_controller = RobotController(robot, dt)
    # Run all tasks and save the output poses
    for task in tasks:
        start = time.time()
        poses = run_task(task, robot_id, robot_controller, dt, fast_eval=fast_eval)
        end = time.time()
        print(f"Task took {end-start:.2f} seconds")
        np.savetxt(os.path.join("task_out", os.path.basename(task)), np.array(poses))

    # Disconnect pybullet
    pb.disconnect()


if __name__ == "__main__":
    main()