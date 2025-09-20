import numpy as np
import roboticstoolbox as rtb

class RobotController:
    def __init__(self, robot: rtb.Robot, dt: float):
        self.robot = robot
        self.dt = float(dt)
        self.home = np.array([2.0, 0.0], dtype=float)

        # task-space tolerance in meters (x–z)
        self.tolerance = 2e-1

        # task-space CLIK (accel-level) gains 
        self.Kp_task = 5000.0  
        self.Kd_task = 2.0 * np.sqrt(self.Kp_task)  

        # Damped least-squares (DLS) parameters
        self.dls_lambda_base = 1e-1
        self.dls_cond_hi = 1e6
        self.dls_lambda_max = 1e0
        self.I2 = np.eye(2)

        # Path state
        self.path: np.ndarray | None = None
        self.x_path: np.ndarray | None = None
        self.x_dot_path: np.ndarray | None = None
        self.x_ddot_path: np.ndarray | None = None
        self.curr_index = 0
        self.max_index = -1
        self.finished = True  # start idle


    # -------------------- Task utilities --------------------

    def dubins_through_poses(self, poses, rho_min, stepsize=0.01):
        """
        Compute a Dubins path through a series of (x, y) poses.
        poses : (N, 2) array of (x, y) waypoints
        rho_min : minimum turning radius
        stepsize : discretization step size
        Returns (P, segments)
        P : (M, 3) array of (x, y, theta) along the path
        segments : list of lengths of each segment
        """
        waypoints = np.array([[p[0], p[1]] for p in poses], dtype=float)
        N = len(waypoints)
        
        assert N >= 1, "Need at least one waypoint"

        # Compute tangent headings
        thetas = np.zeros(N, dtype=float)
        for i in range(N):
            if i == 0:
                dx, dy = (waypoints[1] - waypoints[0]) if N > 1 else (1.0, 0.0)
            elif i == N - 1:
                dx, dy = waypoints[i] - waypoints[i - 1]
            else:
                dx, dy = waypoints[i + 1] - waypoints[i - 1]
            thetas[i] = np.arctan2(dy, dx)

        dub = rtb.DubinsPlanner(curvature=1.0 / rho_min, stepsize=stepsize)
        paths, segments = [], []

        # From home to first waypoint
        start = (float(self.home[0]), float(self.home[1]), thetas[0])
        goal = (waypoints[0, 0], waypoints[0, 1], thetas[0])
        P0, status0 = dub.query(start=start, goal=goal)
        paths.append(np.asarray(P0))
        segments.append(status0.length)

        # Between waypoints
        if N == 1:
            # Single point: just put the waypoint with heading
            paths.append(np.array([[waypoints[0, 0], waypoints[0, 1], thetas[0]]], dtype=float))
            segments.append(0.0)
        else:
            # Multiple waypoints
            for i in range(N - 1):
                start = (waypoints[i, 0], waypoints[i, 1], thetas[i])
                goal = (waypoints[i + 1, 0], waypoints[i + 1, 1], thetas[i + 1])
                Pseg, status = dub.query(start=start, goal=goal)
                if paths:
                    Pseg = Pseg[1:]  # avoid duplicate join
                paths.append(np.asarray(Pseg))
                segments.append(status.length)

        P = np.vstack(paths)
        return P, segments

    # -------------------- Kinematics helpers (x–z plane) --------------------
    #
    def get_direct_kinematics(self, q: np.ndarray) -> np.ndarray:
        """
        Compute end-effector position in x–z plane
        q : (n,) array of joint angles
        returns (2,) array of (x, z) position"""
        T = self.robot.fkine(q)
        return np.asarray(T)[[0, 2], 3].astype(float)  # [x, z]

    def get_cartesian_vel(self, q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        """
        Compute end-effector velocity in x–z plane
        q : (n,) array of joint angles
        q_dot : (n,) array of joint velocities
        returns (2,) array of (x_dot, z_dot) velocity
        """
        J = self.robot.jacob0(q)
        Jxz = np.asarray(J)[[0, 2], :]
        return Jxz @ q_dot

    def get_jdot_qdot(self, q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        """
        Compute end-effector acceleration due to moving Jacobian
        q : (n,) array of joint angles
        q_dot : (n,) array of joint velocities
        returns (2,) array of (x_ddot, z_ddot) acceleration"""
        J_dot = self.robot.jacob0_dot(q, q_dot)
        Jxz_dot = np.asarray(J_dot)[[0, 2], :]
        return Jxz_dot @ q_dot

    # -------------------- Robust Jacobian pseudoinverse --------------------

    def get_jacobian_pseudo_inverse(self, q: np.ndarray, x_vec: np.ndarray) -> np.ndarray:
        """
        Compute joint accelerations from Cartesian accelerations via DLS pseudoinverse
        q : (n,) array of joint angles
        x_vec : (2,) array of desired (x_ddot, z_ddot)
        returns (n,) array of joint accelerations"""
        J = self.robot.jacob0(q)
        Jxz = np.asarray(J)[[0, 2], :]
        if not np.all(np.isfinite(Jxz)):
            return np.zeros(self.robot.n)
        try:
            condJ = np.linalg.cond(Jxz)
        except Exception:
            condJ = np.inf
        lam = (self.dls_lambda_base
               if condJ <= self.dls_cond_hi
               else min(self.dls_lambda_max, self.dls_lambda_base * (condJ / self.dls_cond_hi)))
        JJt = Jxz @ Jxz.T
        JJt = 0.5 * (JJt + JJt.T)  # symmetrize
        try:
            inv = np.linalg.solve(JJt + (lam ** 2) * self.I2, self.I2)
            qdd = Jxz.T @ (inv @ x_vec)
        except np.linalg.LinAlgError:
            qdd = np.zeros(self.robot.n)
        if not np.all(np.isfinite(qdd)):
            qdd = np.zeros(self.robot.n)
        return qdd

    # -------------------- Path preprocessing --------------------

    def get_path_params(self):
        """
        Compute path derivatives (velocity, acceleration) via finite differences
        returns (x, x_dot, x_ddot) each of shape (N, 2)
        """
        x = self.path
        x_dot = np.gradient(x, self.dt, axis=0)
        x_ddot = np.gradient(x_dot, self.dt, axis=0)
        return x, x_dot, x_ddot

    def initialize_new_path(self, path: np.ndarray):
        """
        Initialize a new path for the controller to follow.
        path : (N, 2) array of (x, z) waypoints
        Resets the controller state to start following the new path.
        If path has only one point, the robot will move to that point and stop.
        """

        path = np.asarray(path, dtype=float)
        # Create a smooth Dubins path through the waypoints
        P, _ = self.dubins_through_poses(path[:, 0:2], rho_min=0.05, stepsize=0.1)
        path = P[:, 0:2]

        self.path = path[::2]  # downsample by 2 for speed
        self.curr_index = 0
        self.max_index = len(self.path) - 1
        self.finished = False
        self.x_path, self.x_dot_path, self.x_ddot_path = self.get_path_params()

    def control_step(self, q_current: np.ndarray, dq_current: np.ndarray) -> np.ndarray:
        """
        Compute control torques for the current step.
        q_current : (n,) array of current joint angles
        dq_current : (n,) array of current joint velocities
        returns (n,) array of control torques
        """
        if not self.is_running():
            # Hold with gravity compensation when idle
            return np.squeeze(np.asarray(self.robot.gravload(q_current), dtype=float))

        xe = self.get_direct_kinematics(q_current)
        xe_dot = self.get_cartesian_vel(q_current, dq_current)
        jd = self.get_jdot_qdot(q_current, dq_current)

        # Path tracking / index advance
        err_now = np.linalg.norm(self.path[self.curr_index] - xe)

        # After computing xe:
        if err_now < self.tolerance:
            self.curr_index += 1
        if self.curr_index >= self.max_index:
            self.finished = True


        # Use current (possibly final) target
        target_idx = min(self.curr_index, self.max_index)
        x_err = self.path[target_idx] - xe
        x_dot_err = self.x_dot_path[target_idx] - xe_dot
        a_des = self.x_ddot_path[target_idx] + self.Kd_task * x_dot_err + self.Kp_task * x_err - jd

        # Solve for joint accelerations via DLS pseudoinverse
        y = self.get_jacobian_pseudo_inverse(q_current, a_des)
        # Inverse dynamics to get torques
        tau = self.robot.rne(q_current, dq_current, y)  # inverse dynamics

        return np.squeeze(np.asarray(tau, dtype=float))

    def is_running(self) -> bool:
        return (self.path is not None) and (not self.finished)
