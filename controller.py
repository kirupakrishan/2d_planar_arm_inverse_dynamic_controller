import numpy as np
import roboticstoolbox as rtb
def nonlinear_control(self, q_desired, qd_desired, y):
    q_desired = np.asarray(q).reshape(-1)
    qd_desired = np.asarray(dq).reshape(-1)
    y = np.asarray(y).reshape(-1)

    # Mass/inertia matrix
    B = self.robot.inertia(q)                 # shape (2,2)

    # Coriolis/centrifugal term times qdot
    Cqd = self.robot.coriolis(q_desired, qd_desired) @ qd_desired     # shape (2,)

    # Gravity vector
    g = self.robot.gravload(q_desired)                # shape (2,)

    # Save (optional, if you want to inspect later)
    self.B = B
    self.n = Cqd + g

    # Computed torque
    tau = B @ y + self.n
    return np.squeeze(tau)
