"""
Peter Johnson and Pinky King based on code by
Author: Andrew Q. Pham
Email: apham@g.hmc.edu
Date of Creation: 2/8/20
Description:
    1D Bayes Filter implementation to filter logged x,y,yaw data from a nuscene
    This code is for teaching purposes for HMC ENGR205 System Simulation Lab 2
"""

import csv
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import math


def load_data(filename):
    """Load in the yaw data from the csv log

    Parameters:
    filename (str)  -- the name of the csv log

    Returns:
    yaw_data (float list)   -- the logged yaw data
    """
    f = open(filename)

    file_reader = csv.reader(f, delimiter=',')

    # Load data into dictionary with headers as keys
    # Header: Latitude, Longitude, Time Stamp(ms), ...
    # ..., Yaw(degrees), Pitch(degrees), Roll(degrees)
    data = {}
    header = next(file_reader, None)
    for h in header:
        data[h] = []

    for row in file_reader:
        for h, element in zip(header, row):
            if element in (None,""): 
                continue
            else:
                data[h].append(float(element))

    f.close()

    return data

def sensor_model(data, dt):
    """ Uses stationary car4 data to create a histogram of vehicle
        speed and then create a pdf"""
    x = data["X_4"]
    print(x[0])
    y = data["Y_4"]
    yaw = data["Yaw_4"]
    speed = []
    for i in range(len(x)-2):
        speed.append(math.sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2)/dt)
        print(speed[i])

    plt.figure(1)
    plt.hist(speed, 15, density=True)
    plt.show()

def main():
    """Run a 1D Kalman Filter on logged yaw data from a BNO055 IMU."""

    #filepath = ".\"
    filename = "E205_Lab2_NuScenesData.csv"
    #yaw_data = load_data(filepath + filename)
    
    
    data = load_data(filename)

    #Time step for nuscene data
    dt = 0.5


    sensor_model(data, dt)
    print("Exiting...")

    return 0


if __name__ == "__main__":
    main()
