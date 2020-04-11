import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import math


movement = "01M"
open_path = "data/" + movement + "/" + movement + "_"
save_path = "filtered_data/" + movement + "/" + movement + "_"
f = open(open_path + "encoder.dat", 'r')
f.readline()

with open(save_path + "encoder_filtered.csv", 'w', newline='') as csvfile:
    fieldnames = ['times', 'left', 'right']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    first_read = f.readline()
    time, left, right= first_read.split()
    time = float(time)
    left = float(left)
    right = float(right)
    startingtime = time

    time = time - startingtime
    writer.writerow({'times': time, 'left': left, 'right': right})

    left_prev = left
    right_prev = right

    for line in f:
        time, left, right= line.split()
        time = float(time)
        left = float(left)
        right = float(right)

        time = time - startingtime
        left_d = left - left_prev
        right_d = right - right_prev
        writer.writerow({'times': time, 'left': left_d, 'right': right_d})

        left_prev = left
        right_prev = right


# Load data
f = open(open_path + "lidar.dat", 'r')
f.readline()

with open(save_path + "lidar_filtered.csv", 'w', newline='') as csvfile:
    fieldnames = ['times', 'theta', 'range']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for line in f:
        time, x, y, z, i =line.split()
        time = float(time)
        x = float(x)
        y = float(y)
        z = float(z)

        if (z <= 0.5 and z > -0.5):
            time = time - startingtime
            theta = math.atan2(-x,y)
            range = math.sqrt(x**2 + y**2)
            writer.writerow({'times': time, 'theta': theta, 'range': range})
