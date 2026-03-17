import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

x, y, z = [], [], []

with open('ee_path.csv', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        x.append(float(row['x']))
        y.append(float(row['y']))
        z.append(float(row['z']))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, marker='o', linewidth=1.0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('End-Effector 3D Trajectory')

max_range = max(
    max(x) - min(x),
    max(y) - min(y),
    max(z) - min(z)
) / 2.0
mid_x = (max(x) + min(x)) * 0.5
mid_y = (max(y) + min(y)) * 0.5
mid_z = (max(z) + min(z)) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()