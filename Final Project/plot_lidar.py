import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data
f = open("data/01M/01M_lidar.dat", 'r')

fig = plt.figure(1)
ax = fig.add_subplot(111, projection = '3d')
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])


f.readline()
timePrev = 0
for line in f:
    time, x, y, z, i =line.split()
    time = float(time)
    x = float(x)
    y = float(y)
    z = float(z)

    if time != timePrev:
        ax.clear()

    ax.scatter(x,y,z)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.pause(0.0005)

    timePrev = time

# Load data into dictionary with headers as keys
# Header: Latitude, Longitude, Time Stamp(ms), ...
# ..., Yaw(degrees), Pitch(degrees), Roll(degrees)
# data = {}
# header = next(file_reader, None)
# for h in header:
#     data[h] = []

# for row in file_reader:
#     for h, element in zip(header, row):
#         data[h].append(float(element))

f.close()

v = pptk.viewer(P)
