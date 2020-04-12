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
from PIL import Image

L = 0.545 #m distance between center of Left and right wheels
N = 3
alpha = 1
k = -2
lamda = alpha**2*(N+k)-N
PRINTING = False
PIX_2_M = 7.82
ROWS = 1400
COLS = 5329
ROW_OFFSET_A = 620
COL_OFFSET_A = 730


def load_data(filename):
    """Load data from the csv log

    Parameters:
    filename (str)  -- the name of the csv log

    Returns:
    data (dict)     -- the logged data with data categories as keys
                       and values list of np.doubles
    """
    f = open(filename + ".csv")

    file_reader = csv.reader(f, delimiter=',')

    # Load data into dictionary with headers as keys
    data = {}
    header = next(file_reader, None)
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
                data[h].append(element)

        row_num += 1
    f.close()
    f_log.close()

    return data

def wrap_to_pi(angle):
    """Wrap angle data in radians to [-pi, pi]

    Parameters:
    angle (float)   -- unwrapped angle

    Returns:
    angle (float)   -- wrapped angle
    """
    while angle >= math.pi:
        angle -= 2*math.pi

    while angle <= -math.pi:
        angle += 2*math.pi
    return angle


def propagate_state(x_t_prev, u_t):

    x,y,theta = x_t_prev
    u_l, u_r = u_t

    delta_d = (u_l + u_r)/2
    delta_theta = (u_r - u_l)/L

    x_t_pred = np.array([[x + delta_d*math.cos(theta+delta_theta/2)],
                         [y + delta_d*math.sin(theta+delta_theta/2)],
                         [wrap_to_pi(theta + delta_theta)]], dtype = np.double)


    return x_t_pred

def xy_to_grid(x, y):
    # takes in x and y and returns the row and column of that position in the map
    col = math.floor(x*PIX_2_M) + COL_OFFSET_A
    row = -math.floor(y*PIX_2_M) + ROWS - 1 - ROW_OFFSET_A
    return row, col

def grid_to_xy(row, col):
    # takes in row and column and returns the x, y position of that cell
    x = (col-COL_OFFSET_A)/PIX_2_M
    y = (row - ROWS + 1 + ROW_OFFSET_A)/PIX_2_M
    return x, y

def distance(x1, y1, x2, y2):
    dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return dist


def calc_expected_meas(x_t_pred, azimuth):

    x, y, theta = x_t_pred
    row_start, col_start = get_occupancy_cell(x, y)
    #
    m = math.tan(azimuth)
    col_curr = col_start + 1
    while (MAP[row_curr][col_curr] == 1):
        x_curr, y_curr = grid_to_xy(0, col_curr)
        y_curr = m*x_curr + y
        row_curr, col_curr = xy_to_grid(x_curr, y_curr)

    z_t_pred = distance(x, y, x_curr, y_curr)

    return z_t_pred

def ukf(x_t_prev, sigma_t_prev, u_t, z_t):

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

    filepath = "filtered_data/28M/"
    filename =  "28M_encoder_filtered" #"2020_2_26__17_21_59"
    encoder_data = load_data(filepath + filename)

    u_l = encoder_data["left"]
    u_r = encoder_data["right"]
    times = encoder_data["times"]

    state_estimates = np.empty((N, len(times)))
    x_t_prev = np.array([[0], [0], [math.pi]], dtype = np.double)

    #  Run filter over data
    for t, _ in enumerate(times):

        # Get control input
        u_t = np.array([[u_l[t]], [u_r[t]]], dtype = np.double)
        #print("u_t: ", u_t.shape)

        # Get measurement
        #z_t = np.array()
        #print("z_t: ", z_t.shape)

        # Prediction Step
        x_t = propagate_state(x_t_prev, u_t)
        x_t = x_t.reshape((3,))
        state_estimates[:, t] = x_t

        x_t_prev = x_t

    plt.figure(1)
    plt.scatter(state_estimates[0, :], state_estimates[1, :])
    plt.xlim([-10,1])
    plt.ylim([-5, 6])

    plt.show()

    plt.figure(2)
    plt.scatter(times, state_estimates[2, :])
    plt.scatter(times, u_l, c="r")
    plt.scatter(times, u_r, c="k")
    plt.show()


    filepath = "filtered_data/"
    filename =  "scanPoseEstimates_filtered" #"2020_2_26__17_21_59"
    ground_truth = load_data(filepath + filename)
    x_vals = ground_truth["x_truth"]
    y_vals = ground_truth["y_truth"]

    im = Image.open('map_pic.jpg')
    pixelMap = im.load()
    print(im.size)

    for i in range(len(x_vals)):
        row, col = xy_to_grid(x_vals[i], y_vals[i])
        print(row)
        print(col)
        pixelMap[col, row] = (0,200,0,255)
        pixelMap[col+1, row+1] = (0,200,0,255)
        pixelMap[col+1, row] = (0,200,0,255)
        pixelMap[col+1, row-1] = (0,200,0,255)
        pixelMap[col, row+1] = (0,200,0,255)
        pixelMap[col, row-1] = (0,200,0,255)
        pixelMap[col-1, row+1] = (0,200,0,255)
        pixelMap[col-1, row] = (0,200,0,255)
        pixelMap[col-1, row-1] = (0,200,0,255)

    im.show()
    im.save("out.jpg")
    im.close()



if __name__ == "__main__":
    main()
