import numpy as np
import os

os.makedirs("tasks", exist_ok=True)

# task one is a square of side length 2 centered on (0, 0)
task_one  = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
np.savetxt("tasks/task_1.txt", task_one)

# task two is a circle of radius 1 centered on (0, 0)
theta = np.linspace(0, 2 * np.pi, 100)
task_two = np.array([[np.cos(t), np.sin(t)] for t in theta])
np.savetxt("tasks/task_2.txt", task_two)

# task three is a figure eight offset to (0.5, 0)
t = np.linspace(0, 2 * np.pi, 100)
a = 1.0
b = 0.5
x = a * np.sin(t)
y = b * np.sin(t) * np.cos(t)
task_three = np.array([[x[i] + 0.5, y[i]] for i in range(len(t))])
np.savetxt("tasks/task_3.txt", task_three)