"""
Author: Peter Johnson and Pinky King
Email: pjohnson@g.hmc.edu, pking@g.hmc.edu
Date of Creation: 3/30/20
Description:
    Particle Filter implementation to filtering localization estimate
    This code is for teaching purposes for HMC ENGR205 System Simulation Lab 4
"""

import csv
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import os.path
import scipy as sp
from scipy.stats import norm, uniform
from numpy import linalg as la
from statistics import stdev


HEIGHT_THRESHOLD = 0.0          # meters
GROUND_HEIGHT_THRESHOLD = -.4      # meters
dt = 0.1                        # timestep seconds
X_L = 5.                          # Landmark position in global frame
Y_L = -5.                          # meters
EARTH_RADIUS = 6.3781E6          # meters
NUM_PARTICLES = 20
# variances
VAR_AX = 1.8373
VAR_AY = 1.1991
VAR_THETA = 0.00058709
VAR_LIDAR = 0.0075**2 # this is actually range but works for x and y

PRINTING = False

global V
global VELOCITIES


def load_data(filename):
    """Load data from the csv log

    Parameters:
    filename (str)  -- the name of the csv log

    Returns:
    data (dict)     -- the logged data with data categories as keys
                       and values list of np.doubles
    """
    is_filtered = False
    if os.path.isfile(filename + "_filtered.csv"):
        f = open(filename + "_filtered.csv")
        is_filtered = True
    else:
        f = open(filename + ".csv")

    file_reader = csv.reader(f, delimiter=',')

    # Load data into dictionary with headers as keys
    data = {}
    header = ["X", "Y", "Z", "Time Stamp", "Latitude", "Longitude",
              "Yaw", "Pitch", "Roll", "AccelX", "AccelY", "AccelZ"]
    for h in header:
        data[h] = []

    row_num = 0
    f_log = open("bad_data_log.txt", "w")
    for row in file_reader:
        for h, element in zip(header, row):
            # If got a bad value just use the previous value
            try:
                data[h].append(np.double(element))
            except ValueError:
                data[h].append(data[h][-1])
                f_log.write(str(row_num) + "\n")

        row_num += 1
    f.close()
    f_log.close()

    return data, is_filtered




def ukf(x_t_prev, sigma_t_prev, u_t, z_t)

    # matrix of sigma points
    chi_t_prev = np.empty([N,2*N+1,], dtype=np.double)
    # fill in x_t_prev
    chi_t_prev[:][0] = x_t_prev
    # fill in next N sigma points
    chi_t_prev[:][1:N] = x_t_prev + np.sqrt((N+lamda)*sigma_t_prev)
    # fill in the last N sigma points
    chi_t_prev[:][N+1:2*N] = x_t_prev - np.sqrt((N+lamda)*sigma_t_prev)

    # prediction step
    chi_t_pred = np.empty([N,2*N+1,], dtype=np.double)
    for i in range(0,2*N+1):
        chi_t_pred[:][i] = propagate_state(chi_t_prev[:][i], u_t)

    x_t_pred = (lamda/(lamda+N))*chi_t_pred[:][0]
    for i in range(1, 2*N+1):
        w_i = 1/(2*(N+lamda))
        x_t_pred = x_t_pred + w_i*chi_t_pred[:][i]

    return x_t, sigma_t



def main():
    """Run a EKF on logged data from IMU and LiDAR moving in a box formation around a landmark"""

    #np.random.seed(28)

    filepath = ""
    filename =  "2020_2_26__16_59_7" #"2020_2_26__17_21_59"
    data, is_filtered = load_data(filepath + filename)


    #  Run filter over data
    for t, _ in enumerate(time_stamps):

        # Get control input
        u_t = np.array()
        #print("u_t: ", u_t.shape)

        # Get measurement
        z_t = np.array()
        #print("z_t: ", z_t.shape)

        # Prediction Step
        P_pred_t,  w_tot = prediction_step(P_prev_t, u_t, z_t)

        # Correction Step
        P_t = correction_step(P_pred_t, w_tot)

        #  For clarity sake/teaching purposes, we explicitly update t->(t-1)
        P_prev_t = P_t


if __name__ == "__main__":
    main()
