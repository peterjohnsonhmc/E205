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
alpha = 0.01
beta = 2 # assume Gaussian
k = 0
lamda = alpha**2*(N+k)-N
PRINTING = False
PIX_2_M = 7.82
ROWS = 1400
COLS = 5329
ROW_OFFSET_A = 620
COL_OFFSET_A = 730
R_t = np.array([[0.00435**2,0,0], [0,0.00435**2,0], [0,0,(0.00435/L)**2]])
global MAP

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
                data[h].append(np.float64(element))
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
    while angle > math.pi:
        angle -= 2*math.pi

    while angle <= -math.pi:
        angle += 2*math.pi
    return angle

def wrap_to_pi_sq(angle):
    """Wrap angle data in radians to [-pi, pi]

    Parameters:
    angle (float)   -- unwrapped angle

    Returns:
    angle (float)   -- wrapped angle
    """
    if angle < 0:
        angle = -angle

    while angle > 4*math.pi**2:
        angle -= 4*(math.pi**2)

    return angle


def propagate_state(x_t_prev, u_t):

    x,y,theta = x_t_prev
    u_l, u_r = u_t

    delta_d = (u_l + u_r)/2
    delta_theta = (u_r - u_l)/L

    x_t_pred = np.array([[x + delta_d*math.cos(theta+delta_theta/2)],
                         [y + delta_d*math.sin(theta+delta_theta/2)],
                         [wrap_to_pi(theta + delta_theta)]], dtype = np.float64)

    return x_t_pred.reshape((3,))

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
    print(x,y)
    row_start, col_start = xy_to_grid(x, y)
    # convert from lidar frame to global frame
    azimuth = theta + azimuth - math.pi/2
    m = math.tan(azimuth)
    col_curr = col_start
    row_curr = row_start
    x_curr = x
    y_curr = y
    num = 1
    while (sum(MAP[col_curr,row_curr]) < 20):
        x_curr, y_curr = grid_to_xy(row_curr, col_curr)
        y_curr = m*x_curr + y
        row_curr, col_curr = xy_to_grid(x_curr, y_curr)
        col_curr += 1
        print(num)
        num += 1

    z_t_pred = distance(x, y, x_curr, y_curr)
    print("range", z_t_pred)
    return z_t_pred

def get_chi(x, sigma):

    chi = np.empty([N,2*N+1], dtype=np.float64)
    # fill in x
    x = x.reshape((3,))
    chi[:,0] = x
    # fill in next N sigma points
    chi[:,1:N+1] = np.array([[x[0], x[0], x[0]], [x[1], x[1], x[1]], [x[2], x[2], x[2]]]) + np.sqrt((N+lamda)*sigma)
    # fill in the last N sigma points
    chi[:,N+1:2*N+1] = np.array([[x[0], x[0], x[0]], [x[1], x[1], x[1]], [x[2], x[2], x[2]]]) - np.sqrt((N+lamda)*sigma)
    chi[2,0] = wrap_to_pi(chi[2,0])
    chi[2,1] = wrap_to_pi(chi[2,1])
    chi[2,2] = wrap_to_pi(chi[2,2])
    chi[2,3] = wrap_to_pi(chi[2,3])
    chi[2,4] = wrap_to_pi(chi[2,4])
    chi[2,5] = wrap_to_pi(chi[2,5])
    chi[2,6] = wrap_to_pi(chi[2,6])

    return chi

def prediction_step(x_t_prev, sigma_t_prev, u_t):

    chi_t_prev = get_chi(x_t_prev, sigma_t_prev)
    # prediction step
    chi_t_prop = np.empty([N,2*N+1], dtype=np.float64)
    for i in range(0,2*N+1):
        chi_t_prop[:,i] = propagate_state(chi_t_prev[:,i], u_t)

    # calculate the predicted x
    x_t_pred = (lamda/(lamda+N))*chi_t_prop[:,0]
    for i in range(1, 2*N+1):
        w_m_i = 1/(2*(N+lamda))
        x_t_pred = x_t_pred + w_m_i*chi_t_prop[:,i]
    x_t_pred[2]= wrap_to_pi(x_t_pred[2])

    # calculate the predicted
    col = chi_t_prop[:,0] - x_t_pred
    col[2] = wrap_to_pi(col[2])
    sigma_t_pred = (lamda/(N+lamda) + (1-alpha**2+beta))*col.dot(np.transpose(col))
    for i in range(1, 2*N+1):
        w_c_i = 1/(2*(N+lamda))
        col = chi_t_prop[:,i] - x_t_pred
        col[2] = wrap_to_pi(col[2])
        sigma_t_pred += w_c_i*col.dot(np.transpose(col))
    sigma_t_pred += R_t # don't forget to add in Rt
    sigma_t_pred[2,0] = wrap_to_pi_sq(sigma_t_pred[2,0])
    sigma_t_pred[2,1] = wrap_to_pi_sq(sigma_t_pred[2,1])
    sigma_t_pred[2,2] = wrap_to_pi_sq(sigma_t_pred[2,2])
    print(sigma_t_pred)

    return x_t_pred, sigma_t_pred


def correction_step(x_t_pred, sigma_t_pred, z_t, Q_t):

    # get the predicted chi from the predicted sigma and x
    chi_t_pred = get_chi(x_t_pred, sigma_t_pred)

    # get the matrix of predicted measurements
    Z_t_pred = np.empty([1,2*N+1,], dtype=np.float64)
    for i in range(0, 2*N+1):
        Z_t_pred[:,i] = calc_expected_meas(chi_t_pred[:,i], z_t[1])

    # consolidate into one predicted measurements
    z_t_pred = (lamda/(lamda+N))*Z_t_pred[:,0]
    for i in range(1, 2*N+1):
        w_m_i = 1/(2*(N+lamda))
        z_t_pred = z_t_pred + w_m_i*Z_t_pred[:,i]

    # Update S
    col = Z_t_pred[:,0] - z_t_pred
    S_t =  (lamda/(N+lamda) + (1-alpha**2+beta))*col.dot(np.transpose(col))
    for i in range(1, 2*N+1):
        w_c_i = 1/(2*(N+lamda))
        col = Z_t_pred[:,i] - z_t_pred
        S_t += w_c_i*col.dot(np.transpose(col))
    S_t += Q_t

    # Calculate sigma_x_z_t
    col_chi = (chi_t_pred[:,0] - x_t_pred).reshape((3,1))
    col_z = (Z_t_pred[:,0] - z_t_pred).reshape((1,1))
    sigma_x_z_t =  (lamda/(N+lamda) + (1-alpha**2+beta))*col_chi*col_z
    for i in range(1, 2*N+1):
        w_c_i = 1/(2*(N+lamda))
        col_chi = (chi_t_pred[:,i] - x_t_pred).reshape((3,1))
        col_z = (Z_t_pred[:,i] - z_t_pred).reshape((1,1))
        sigma_x_z_t += w_c_i*col_chi*col_z


    # Calculate Kalman gain
    K_t = sigma_x_z_t/S_t
    # Update estimate of mu
    x_t_est = x_t_pred + K_t.dot(z_t[0] - z_t_pred)
    # Update sigma_t_est
    sigma_t_est = sigma_t_pred + K_t.dot(S_t).dot(np.transpose(K_t))

    return x_t_est, sigma_t_est

def plot_pixel(imMap, x, y, color):
    row, col = xy_to_grid(x, y)
    imMap[col, row] = color
    imMap[col+1, row+1] = color
    imMap[col+1, row] = color
    imMap[col+1, row-1] = color
    imMap[col, row+1] = color
    imMap[col, row-1] = color
    imMap[col-1, row+1] = color
    imMap[col-1, row] = color
    imMap[col-1, row-1] = color
    return 0

def main():
    """Run a EKF on logged data from IMU and LiDAR moving in a box formation around a landmark"""

    #np.random.seed(28)

    im = Image.open('map_pic.jpg')
    global MAP
    MAP = im.load()

    filepath = "filtered_data/01M/"
    filename =  "01M_encoder_filtered" #"2020_2_26__17_21_59"
    encoder_data = load_data(filepath + filename)

    filename =  "01M_lidar_gauss" #"2020_2_26__17_21_59"
    lidar_data = load_data(filepath + filename)

    u_l = encoder_data["left"]
    u_r = encoder_data["right"]
    encoder_times = encoder_data["times"]

    z_range_mu = lidar_data["range_mu"]
    z_range_var = lidar_data["range_var"]
    z_azimuth_mu = lidar_data["theta_mu"]
    lidar_times = lidar_data["times"]

    state_estimates = np.empty((N, len(encoder_times)))
    x_t_prev = np.array([[0], [0], [0]], dtype = np.float64)
    sigma_t_prev = R_t

    #  Run filter over data
    lidar_index = 0
    for t, _ in enumerate(encoder_times):

        print(encoder_times[t])

        # Get control input
        u_t = np.array([[u_l[t]], [u_r[t]]], dtype = np.float64)
        # print("u_t: ", u_t.shape)

        # figure out if we will correct
        if (lidar_times[lidar_index] > encoder_times[t] and lidar_times[lidar_index] < encoder_times[t+1]):
            x_t_pred, sigma_t_pred = prediction_step(x_t_prev, sigma_t_prev, u_t)
            print("correcting")
            # Get measurement
            print(type(z_range_mu[lidar_index]))
            z_t = np.array([[z_range_mu[lidar_index]], [z_azimuth_mu[lidar_index]]], dtype = np.float64)
            Q_t = z_range_var[lidar_index]
            # print("z_t: ", z_t.shape)
            x_t_est, sigma_t_est = correction_step(x_t_pred, sigma_t_pred, z_t, Q_t)
            print("corrected sigma:", sigma_t_est)
            lidar_index += 1
        else:
            # we can't predict without correcting
            x_t_pred = propagate_state(x_t_prev, u_t)
            x_t_est = x_t_pred
            sigma_t_est = R_t + sigma_t_prev

        state_estimates[:, t] = x_t_est
        x_t_prev = x_t_est
        sigma_t_prev = sigma_t_est



    plt.figure(1)
    plt.scatter(state_estimates[0, :], state_estimates[1, :])
    plt.xlim([-10,1])
    plt.ylim([-5, 6])

    plt.show()

    plt.figure(2)
    plt.scatter(encoder_times, state_estimates[2, :])
    plt.scatter(encoder_times, u_l, c="r")
    plt.scatter(encoder_times, u_r, c="k")
    plt.show()


    filepath = "filtered_data/"
    filename =  "scanPoseEstimates_filtered" #"2020_2_26__17_21_59"
    ground_truth = load_data(filepath + filename)
    x_vals = ground_truth["x_truth"]
    y_vals = ground_truth["y_truth"]

    im = Image.open('map_pic.jpg')
    pixelMap = im.load()

    for i in range(len(state_estimates[0,:])):
        x = state_estimates[0,i]
        y = state_estimates[1,i]
        plot_pixel(pixelMap,x,y,(0,200,200,255))


    for i in range(len(x_vals)):
        plot_pixel(pixelMap,x_vals[i],y_vals[i],(0,200,0,255))



    im.show()
    im.save("out.jpg")
    im.close()



if __name__ == "__main__":
    main()
