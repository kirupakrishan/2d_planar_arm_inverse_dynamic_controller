class RobotController:
    def __init__(self, robot, dt):
        self.robot = robot
        self.robot: rtb.Robot
        self.dt = dt

        #Dyanmic parameters
        self.B = np.array([[0, 0], [0, 0]])  # Inertia matrix
        self.C = np.array([[0, 0], [0, 0]])  # Coriolis matrix
        self.F = np.array([[0, 0], [0, 0]])  # Friction matrix
        self.g = np.array([0, 0])  # Gravity vector  



        # Task-space gains (for desired cart vel from error)
        self.Kp_task = 10.0    # higher so small cart errors create noticeable command
        self.Kd_task = 5.0

        # Joint-space PD on velocities (acts on vel error -> torque)
        self.Kp_joint = 10.0
        self.Kd_joint = 5.0

        # tolerances and safety
        self.pos_tol = 5e-3    # consider waypoint reached if < 5 mm
        self.torque_saturate = 80.0

        # path state (set in initialize_new_path)
        self.path = None
        self.curr_index = 0
        self.max_index = -1
        self.prev_q = np.array([0.0, 0.0])
        self.prev_q_dot = np.array([0.0, 0.0])
        self.q_dot_dot = np.array([0.0, 0.0])

    def initialize_new_path(self, path: np.ndarray):
        path = np.asarray(path)
        if path.ndim == 1:
            path = path.reshape(1, -1)
        self.path = path
        self.curr_index = 0
        self.max_index = len(path) - 1

        # simple heuristic: dense if > 20 points
        self.dense_path = len(path) > 20

    def control_step(self, q_current, dq_current):
        # Finished?
        if not self.is_running():
            return np.zeros(2)

        # Current EE pose (x, z)
        ee = self.robot.fkine(q_current)
        ee_pos = np.asarray(ee)[0:3:2, 3].astype(float)

        # Current target
        target = self.path[self.curr_index]
        e = target - ee_pos

        # For dense paths, also compute feedforward cartesian velocity
        next_idx = min(self.curr_index + 1, self.max_index)
        next_target = self.path[next_idx]
        cart_vel_ff = (next_target - target) / self.dt if self.dense_path else np.zeros(2)

        # Desired cartesian velocity (PD in task space + feedforward if dense)
        v_des_cart = self.Kp_task * e + self.Kd_task * cart_vel_ff

        # Jacobian (take x,z rows)
        J = self.robot.jacob0(q_current)
        Jxz = np.asarray(J)[0:3:2, :]

        # Desired joint velocities from inverse differential kinematics
        try:
            J_pinv = np.linalg.pinv(Jxz, rcond=1e-3)
        except Exception:
            J_pinv = np.zeros((2, 2))
        vel_des_q = J_pinv @ v_des_cart

        # Velocity error
        vel_err = vel_des_q - np.asarray(dq_current)

        # Joint-space PD for torque
        torque = self.Kp_joint * vel_err - self.Kd_joint * np.asarray(dq_current)

        # Saturate torque
        torque = np.clip(torque, -self.torque_saturate, self.torque_saturate)

        # Waypoint advancement logic
        if self.dense_path:
            # stream waypoints for dense trajectories
            if self.curr_index < self.max_index:
                self.curr_index += 1
        else:
            # hold waypoint until reached for sparse trajectories
            if np.linalg.norm(e) < self.pos_tol and self.curr_index < self.max_index:
                self.curr_index += 1

        # Gravity compensation
        gravload = -self.robot.gravload(q_current)

        return np.squeeze(torque) + gravload

    def is_running(self):
        return self.curr_index < self.max_index