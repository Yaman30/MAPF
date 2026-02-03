import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

#Referens punkter
waypoints = np.array([
    [0, 0],
    [5, 0],
    [10, 7],
    [13, 5],
    [16, 5],
    [20, 8],
    [25, -3]
])

# Robotens
x = 0.0
y = 0.0
theta = 0.0
speed = 0.2
L = 1.0
W = 0.5

# Kartan
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(-5, 30)
ax.set_ylim(-5, 10)
ax.set_aspect('equal')
ax.grid(True)

# Hindren
obstacle = patches.Rectangle((8, -2), 4, 4, color='brown', label='Hinder')
ax.add_patch(obstacle)

# Banan
'''
for p in waypoints:
    ax.plot(p[0], p[1], 'rx')
'''
ax.plot(waypoints[:, 0], waypoints[:, 1], 'k--', alpha=0.2, linewidth=5, label='bana')
ax.plot(waypoints[:, 0], waypoints[:, 1], 'mp', label='Waypoints')

# Roboten som är en Rektangel
robot_rect = patches.Rectangle((0, 0), L, W, color='cyan', alpha=0.5, label='Robot')
ax.add_patch(robot_rect)

ax.legend()
plt.show()