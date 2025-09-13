import os
import numpy as np
import matplotlib.pyplot as plt

tasks = [os.path.join("tasks", f) for f in os.listdir("tasks")]

task_out = [os.path.join("task_out", f) for f in os.listdir("task_out")]

import numpy as np

def fk_2R(q, l1, l2):
    q1 = q[:, 0]
    q2 = q[:, 1]
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    z = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    return np.array([x, z])


for task, task_out in zip(tasks, task_out):
    data = np.loadtxt(task)
    x_in = data[:, 0]
    y_in = data[:, 1]

    output = np.loadtxt(task_out)
    x_out = fk_2R(output, 1, 1)[0, :]
    y_out = fk_2R(output, 1, 1)[1, :]

     # Input path
    plt.scatter(x_in, y_in, color='blue', label="Input points")
    plt.plot(x_in, y_in, linestyle='--', color='blue', alpha=0.5)

    # Output path
    plt.scatter(x_out, y_out, color='green', label="Output points")
    plt.plot(x_out, y_out, linestyle='--', color='green', alpha=0.5)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(task_out)
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.show()