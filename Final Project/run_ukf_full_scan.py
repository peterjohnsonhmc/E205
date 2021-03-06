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


global ax
L = 0.545 #m distance between center of Left and right wheels
N = 3
alpha = .01
beta = alpha**2-1 # assume Gaussian
beta = -0.9
k = 0
lamda = alpha**2*(N-k)
#lamda = -2
PRINTING = False
PIX_2_M = 7.605 #7.82
ROWS = 1395 #1492
COLS = 5329 #5352
ROW_OFFSET_A = 785 # 835
COL_OFFSET_A = 745 # 760
R_t = np.array([[0.014**2,0,0], [0,0.027**2,0], [0,0,0.02**2]]) # 0.02 is good for A to B
#R_t = np.array([[0.00435**2,0,0], [0,0.00435**2,0], [0,0,(0.00435/L)**2]])
global MAP
global plottingMAP
global im2

#NUmber of beams per time step
NUM_BEAMS = 5

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

    return x_t_pred.reshape((N,))

def calc_prop_jacobian_x(x_t_prev, u_t):

    x,y,theta = x_t_prev
    u_l, u_r = u_t

    delta_d = (u_l + u_r)/2
    delta_theta = (u_r - u_l)/L

    G_x_t = np.array([[1, 0, -delta_d*math.sin(theta+delta_theta/2)],
                         [0, 1,  delta_d*math.cos(theta+delta_theta/2)],
                         [0, 0,  1]], dtype = np.float64)
    return G_x_t

def calc_prop_jacobian_u(x_t_prev, u_t):
    x,y,theta = x_t_prev
    u_l, u_r = u_t

    delta_d = (u_l + u_r)/2
    delta_theta = (u_r - u_l)/L

    G_u_t = np.array([[(1/2)*math.cos(theta+delta_theta/2)-(1/L)*delta_d*math.sin(theta+delta_theta/2), (1/2)*math.cos(theta+delta_theta/2)+(1/L)*delta_d*math.sin(theta+delta_theta/2)],
                         [(1/2)*math.sin(theta+delta_theta/2)+(1/L)*delta_d*math.cos(theta+delta_theta/2), (1/2)*math.sin(theta+delta_theta/2)-(1/L)*delta_d*math.cos(theta+delta_theta/2)],
                         [1/L, -1/L]], dtype = np.float64)

    return G_u_t

def xy_to_grid(x, y):
    # takes in x and y and returns the row and column of that position in the map
    col = math.floor(x*PIX_2_M) + COL_OFFSET_A
    row = -math.floor(y*PIX_2_M) + ROW_OFFSET_A
    return row, col

def grid_to_xy(row, col):
    # takes in row and column and returns the x, y position of that cell
    x = (col-COL_OFFSET_A)/PIX_2_M
    y = -(row - ROW_OFFSET_A)/PIX_2_M
    return x, y

def distance(x1, y1, x2, y2):
    dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return dist


def calc_expected_meas(x_t_pred, azimuth_t):
    #print("Sigma point: ", x_t_pred)
    x, y, theta = x_t_pred
    z_t_pred = np.zeros((NUM_BEAMS, 1))

    for i in range(NUM_BEAMS):

        azimuth = azimuth_t[i]
        # convert from lidar frame to global frame
        azimuth = wrap_to_pi(theta + azimuth)
        #print("Azimuth :", azimuth)
        m = math.tan(azimuth)
        x_curr = x
        y_curr = y
        row_curr, col_curr = xy_to_grid(x, y)
        if (abs(m) > 1):
            m = 1/m # in the case that we flip x and y
            flipped_azimuth = wrap_to_pi(-(azimuth-math.pi/2))
            # because of how y is indexed, we want to decrease the col to move up
            # and increase the col to move down
            if (flipped_azimuth > math.pi/2 or flipped_azimuth < -math.pi/2):
                increment = 1
            else:
                increment = -1
            while (in_bounds(row_curr, col_curr)==True and sum(MAP[col_curr,row_curr]) < 300):
                row_curr = row_curr+increment
                x_curr, y_curr = grid_to_xy(row_curr, col_curr)
                x_curr = m*y_curr + x
                _, col_curr = xy_to_grid(x_curr, y_curr)
                #ax.scatter(x_curr, y_curr, s=2, marker='.', color='r', alpha=0.2)

        else:
            if (azimuth > math.pi/2 or azimuth < -math.pi/2):
                increment = -1
            else:
                increment = 1
            while (in_bounds(row_curr, col_curr)==True and sum(MAP[col_curr,row_curr]) < 300):
                col_curr = col_curr+increment
                x_curr, y_curr = grid_to_xy(row_curr, col_curr)
                #print(x_curr,y_curr)
                y_curr = m*x_curr + y
                row_curr, _ = xy_to_grid(x_curr, y_curr)
                #ax.scatter(x_curr, y_curr, s=2, marker='.', color='r', alpha=0.2)
                #print(col_curr, row_curr)

        z_t_pred[i] = distance(x, y, x_curr, y_curr)
    #print("range_pred: ", z_t_pred)
    return z_t_pred

def in_tunnel(sigma_point, mu):
    # print("in in_tunnel function")
    x, y, theta = sigma_point
    x_m, y_m, theta_m = mu
    row_curr, col_curr = xy_to_grid(x, y)
    if (sum(MAP[col_curr,row_curr]) < 300):
        return sigma_point
    else:
        x_curr, y_curr = x, y
        # first, try moving up/down
        m = 0 #(x_m-x)/(y_m-y) # in the case that we flip x and y
        azimuth = wrap_to_pi(math.atan2(y_m-y, x_m-x))
        flipped_azimuth = wrap_to_pi(-(azimuth-math.pi/2))
        # because of how y is indexed, we want to decrease the col to move up
        # and increase the col to move down
        if (y_m > y):
            increment = -1
        else:
            increment = 1
        while (in_bounds(row_curr, col_curr)==False or sum(MAP[col_curr,row_curr]) > 300):
            row_curr = row_curr+increment
            x_curr, y_curr = grid_to_xy(row_curr, col_curr)
            x_curr = m*y_curr + x
            _, col_curr = xy_to_grid(x_curr, y_curr)
            #ax.scatter(x_curr, y_curr, s=2, marker='.', color='r', alpha=0.2)
            if (row_curr < -500) or (row_curr > 500+ROWS):
                break
            print("y", y, y_m, increment, m, azimuth, row_curr, col_curr)
        # move robot a bit more in bounds so not just on edge
        if (increment == -1):
            adjustment = 2
        else:
            adjustment = -2.5
        x_vert = x_curr
        y_vert = y_curr + adjustment

        # try moving left/right
        m = 0 #(y_m-y)/(x_m-x)
        azimuth = math.atan2(y_m-y, x_m-x)
        if (x_m > x):
            increment = 1
        else:
            increment = -1
        while (in_bounds(row_curr, col_curr)==False or sum(MAP[col_curr,row_curr]) > 300):
            col_curr = col_curr+increment
            x_curr, y_curr = grid_to_xy(row_curr, col_curr)
            #print(x_curr,y_curr)
            y_curr = m*x_curr + y
            row_curr, _ = xy_to_grid(x_curr, y_curr)
            #ax.scatter(x_curr, y_curr, s=2, marker='.', color='r', alpha=0.2)
            #print(col_curr, row_curr)
            if (col_curr < -500) or (col_curr > 500+COLS):
                break
            print("x", x, x_m, increment, m, azimuth, row_curr, col_curr)
        # move robot a bit more in bounds so not just on edge
        if (increment == -1):
            adjustment = -2.5
        else:
            adjustment = 2
        y_hor = y_curr
        x_hor = x_curr + adjustment

        if (distance(x,y,x_vert,y_vert) < distance(x,y,x_hor,y_hor)):
            x_curr, y_curr = x_vert, y_vert
        else:
            x_curr, y_curr = x_hor, y_hor
        sigma_point[0] = x_curr
        sigma_point[1] = y_curr
        return sigma_point

def in_bounds(row, col):
    if (row < ROWS and row >= 0 and col < COLS and col >= 0):
        return True
    else:
        return False

def get_chi(x, sigma, x_t_prev):

    #L = np.linalg.cholesky(sigma)

    chi = np.empty([N,2*N+1], dtype=np.float64)
    # fill in x
    x = x.reshape((N,))
    # chi[:,0] = x
    # # fill in next N sigma points
    # print("sigma_t_prev: ", sigma)
    # x_array = np.array([[x[0], x[0], x[0]], [x[1], x[1], x[1]], [x[2], x[2], x[2]]])
    sqrt_sigma = np.sqrt((N+lamda)*sigma)
    # chi[:,1:N+1] = np.add(x_array, sqrt_sigma)
    # print(x_array)
    # print(sqrt_sigma)
    # print(x_array.shape)
    # print(sqrt_sigma.shape)
    # # fill in the last N sigma points
    # chi[:,N+1:2*N+1] = np.array([[x[0], x[0], x[0]], [x[1], x[1], x[1]], [x[2], x[2], x[2]]]) - np.sqrt((N+lamda)*sigma)

    chi[:,0] = x
    for i in range(1,N+1):
        chi[:,i] = x + sqrt_sigma[:,i-1]
    for i in range(N+1,2*N+1):
        chi[:,i] = x - sqrt_sigma[:,i-(N+1)]


    chi[2,0] = wrap_to_pi(chi[2,0])
    chi[2,1] = wrap_to_pi(chi[2,1])
    chi[2,2] = wrap_to_pi(chi[2,2])
    chi[2,3] = wrap_to_pi(chi[2,3])
    chi[2,4] = wrap_to_pi(chi[2,4])
    chi[2,5] = wrap_to_pi(chi[2,5])
    chi[2,6] = wrap_to_pi(chi[2,6])

    corrector = chi[:,0]
    row, col = grid_to_xy(chi[0,0], chi[1,0])
    if (in_bounds(row, col)==False or sum(MAP[col,row]) > 300):
        corrector = x_t_prev
    for i in range(1,2*N+1):
        chi[:,i] = in_tunnel(chi[:,i], corrector)
    chi[:,0] = corrector.reshape((3,))

    return chi

def prediction_step(x_t_prev, sigma_t_prev, u_t):

    print("u_t: ", u_t)
    print("x_t_prev: ", x_t_prev)

    chi_t_prev = get_chi(x_t_prev, sigma_t_prev, x_t_prev)
    # prediction step
    chi_t_prop = np.empty([N,2*N+1], dtype=np.float64)
    for i in range(0,2*N+1):
        chi_t_prop[:,i] = propagate_state(chi_t_prev[:,i], u_t)

    # calculate the predicted x
    x_t_pred = (lamda/(lamda+N))*chi_t_prop[:,0]
    # i = 0
    # row, col = xy_to_grid(chi_t_prop[0,0], chi_t_prop[1,0])
    # while (in_bounds(row, col)==False or sum(MAP[col,row]) > 300):
    #     print("BAD MEASUREMENT")
    #     chi_t_prop[:,0] = chi_t_prop[:,i]
    #     i += 1
    #     if i == (2*N+1):
    #         break
    #     row, col = xy_to_grid(chi_t_prop[0,i], chi_t_prop[1,i])

    # for i in range(1, 2*N+1):
    #     row, col = xy_to_grid(chi_t_prop[0,i], chi_t_prop[1,i])
    #     if (in_bounds(row, col)==False or sum(MAP[col,row]) > 300):
    #         chi_t_prop[:,i] = chi_t_prop[:,0]
    for i in range(1, 2*N+1):
        w_m_i = 1/(2*(N+lamda))
        x_t_pred = x_t_pred + w_m_i*chi_t_prop[:,i]
    x_t_pred[2]= wrap_to_pi(x_t_pred[2])

    # calculate the predicted
    col = chi_t_prop[:,0] - x_t_pred
    col[2] = wrap_to_pi(col[2])
    sigma_t_pred = (lamda/(N+lamda) + (1-alpha**2+beta))*col.dot(np.transpose(col))
    w_c_i = 1/(15*(N+lamda))
    for i in range(1, 2*N+1):
        col = chi_t_prop[:,i] - x_t_pred
        col[2] = wrap_to_pi(col[2])
        sigma_t_pred += w_c_i*col.dot(np.transpose(col))
    sigma_t_pred += R_t # don't forget to add in Rt
    print("sigma_t_pred: ", sigma_t_pred)
    sigma_t_pred[2,0] = min(wrap_to_pi_sq(sigma_t_pred[2,0]), 0.175**2)
    sigma_t_pred[2,1] = min(wrap_to_pi_sq(sigma_t_pred[2,1]), 0.175**2)
    sigma_t_pred[2,2] = min(wrap_to_pi_sq(sigma_t_pred[2,2]), 0.175**2)

    x_t_pred = x_t_pred.reshape((3,1))
    return x_t_pred, sigma_t_pred


def correction_step(x_t_pred, sigma_t_pred, z_t, azimuth_t, Q_t, x_t_prev):

    #print("range: ", z_t[0])
    #z_t = z_t.reshape((NUM_BEAMS,1))

    # get the predicted chi from the predicted sigma and x
    chi_t_pred = get_chi(x_t_pred, sigma_t_pred, x_t_prev)
    #print("chi_t_pred: ", chi_t_pred)
    # for i in range(1, 2*N+1):
    #     row, col = xy_to_grid(chi_t_pred[0,i], chi_t_pred[1,i])
    #     if (in_bounds(row, col)==False or sum(MAP[col,row]) > 300):
    #         chi_t_pred[:,i] = chi_t_pred[:,0]


    # get the matrix of predicted measurements
    Z_t_pred = np.empty((NUM_BEAMS,2*N+1), dtype=np.float64)

    for i in range(0, 2*N+1):
        Z_t_pred[:,i] = calc_expected_meas(chi_t_pred[:,i], azimuth_t).reshape((NUM_BEAMS,))

    # consolidate into one predicted measurements
    z_t_pred = (lamda/(lamda+N))*Z_t_pred[:,0]
    w_m_i = 1/(2*(N+lamda))
    for i in range(1, 2*N+1):
        z_t_pred = z_t_pred + w_m_i*Z_t_pred[:,i]
    z_t_pred = z_t_pred.reshape((NUM_BEAMS,1))

    # Update S
    col = (Z_t_pred[:,0].reshape((NUM_BEAMS,1)) - z_t_pred) #.reshape((NUM_BEAMS,1))
    S_t =  (lamda/(N+lamda) + (1-alpha**2+beta))*col.dot(np.transpose(col))
    for i in range(1, 2*N+1):
        w_c_i = 1/(2*(N+lamda))
        col = Z_t_pred[:,i].reshape((NUM_BEAMS, 1)) - z_t_pred
        S_t += w_c_i*col.dot(np.transpose(col))
    S_t += Q_t


    # Calculate sigma_x_z_t
    col_chi = (chi_t_pred[:,0].reshape((N,1)) - x_t_pred).reshape((N,1))
    col_z = (Z_t_pred[:,0].reshape((NUM_BEAMS,1)) - z_t_pred).reshape((NUM_BEAMS,1))
    sigma_x_z_t =  (lamda/(N+lamda) + (1-alpha**2+beta))*col_chi.dot(col_z.T)
    for i in range(1, 2*N+1):
        w_c_i = 1/(2*(N+lamda))
        col_chi = (chi_t_pred[:,i].reshape((N,1)) - x_t_pred).reshape((N,1))
        col_z = (Z_t_pred[:,i].reshape((NUM_BEAMS,1)) - z_t_pred).reshape((NUM_BEAMS,1))
        sigma_x_z_t += w_c_i*col_chi.dot(col_z.T)

    # Calculate Kalman gain
    K_t = sigma_x_z_t.dot(np.linalg.inv(S_t))
    # Update estimate of mu
    residual = (z_t - z_t_pred).reshape((NUM_BEAMS,1))
    percent_off = 100*np.min(residual)/np.mean(z_t)
    #if abs(percent_off) > 30:
    #    x_t_est = x_t_pred
    #else:
    x_t_est = x_t_pred.reshape((N,1)) + K_t.dot(residual)
    row, col = xy_to_grid(x_t_est[0], x_t_est[1])
    if (in_bounds(row, col)==False or sum(MAP[col,row]) > 300):
        x_t_est = x_t_pred
    # Update sigma_t_est
    #print("delta_sigma:", K_t.dot(S_t).dot(np.transpose(K_t)))
    sigma_t_est = sigma_t_pred - K_t.dot(S_t).dot(np.transpose(K_t))
    for i in range(N):
        for j in range(N):
            if sigma_t_est[i,j] < 0:
                sigma_t_est[i,j] = 1e-20

    return x_t_est, sigma_t_est

def plot_1_pixel(imMap, x, y, color):
    row, col = xy_to_grid(x, y)
    imMap[col, row] = color
    return 0

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


def draw_points(range, azimuth, x_off, y_off, theta):
    x = 0#x_off
    y = 0#y_off
    row_start, col_start = xy_to_grid(x, y)
    # convert from lidar frame to global frame
    azimuth = wrap_to_pi(theta + azimuth)
    m = math.tan(azimuth)
    if (azimuth > math.pi/2 or azimuth < -math.pi/2):
        increment = -1
    else:
        increment = 1
    col_curr = col_start
    row_curr = row_start
    x_curr = x
    y_curr = y
    dist = 0
    while (dist < range):
        col_curr = col_curr+increment
        x_curr, y_curr = grid_to_xy(row_curr, col_curr)
        #print(x_curr,y_curr)
        y_curr = m*x_curr + y
        row_curr, _ = xy_to_grid(x_curr, y_curr)
        ax.scatter(x_curr+x_off, y_curr+y_off,  s=2, marker='.', color='g')
        dist = distance(x, y, x_curr, y_curr)
    return 0

def main():
    """Run a EKF on logged data from IMU and LiDAR moving in a box formation around a landmark"""

    #np.random.seed(28)
    global NUM_BEAMS

    im = Image.open('map.png')
    global MAP
    MAP = im.load()

    global im2
    im2 = Image.open('map.png')
    global plottingMAP
    plottingMAP = im2.load()

    filepath = "filtered_data/"
    filename =  "scanPoseEstimates_filtered" #"2020_2_26__17_21_59"
    ground_truth = load_data(filepath + filename)
    stops = ground_truth["pose_num"]
    x_vals = ground_truth["x_truth"]
    y_vals = ground_truth["y_truth"]

    plt.figure(0)
    # plt.subplot(2,1,1)
    global ax
    fig, ax = plt.subplots()
    low_x, low_y = grid_to_xy(0,0)
    high_x, high_y = grid_to_xy(ROWS-1,COLS-1)
    ax.imshow(im2, extent=[low_x, high_x, high_y, low_y ])
    plt.xlim([-100,100])
    plt.ylim([high_y, low_y])

    movements = ["01M", "02M", "03M", "04M", "05M", "06M", "07M", "08M", "09M", "10M", "11M", "12M", "13M", "14M", "15M", "16M"]
    x_t_prev = np.array([[0], [0], [0]], dtype = np.float64)
    sigma_t_prev = R_t
    x_estimates = []
    y_estimates = []
    theta_estimates = []

    state_estimates_encoder = []

    for movement in movements:

        filepath = "filtered_data/" + movement + "/"
        filename =  movement + "_encoder_filtered" #"2020_2_26__17_21_59"
        encoder_data = load_data(filepath + filename)

        filename =  movement + "_lidar_multi_" + str(5) #"2020_2_26__17_21_59"
        lidar_data = load_data(filepath + filename)

        u_l = encoder_data["left"]
        u_r = encoder_data["right"]
        encoder_times = encoder_data["times"]

        z_range_mu = lidar_data["range_mu"]
        z_range_var = lidar_data["range_var"]
        z_azimuth_mu = lidar_data["theta_mu"]
        lidar_times = lidar_data["times"]

        num = movement[0:2]
        for i in range(len(stops)):
            if (stops[i])[0:2] == num:
                break

        x_truth = x_vals[i]
        y_truth = y_vals[i]


        # x_t_pred = np.array([[0], [0], [0]], dtype = np.float64)
        # x_t_pred = x_t_pred.reshape((3,))
        # #plot_1_pixel(plottingMAP, x_t_pred[0], x_t_pred[1], (0,200,200,255))
        # sigma_t_pred = R_t*10
        # z_t = np.array([[1.416258292], [-2.654795948]], dtype = np.float64)
        # Q_t = 0.01
        # x_t_est, sigma_t_est = correction_step(x_t_pred, sigma_t_pred, z_t, Q_t)
        # plot_1_pixel(plottingMAP, x_t_est[0], x_t_est[1], (0,200,0,255))
        # print("x_t_est: ", x_t_est)
        # print("sigma_t_est: ", sigma_t_est)
        # im2.show()
        # im2.save("out.png")
        # im2.close()

        # Run filter over data
        lidar_index = 1


        # plt.scatter(251.2194, 64.61)
        # plt.show()

        # plt.subplot(2,1,2)
        # plt.xlim([55, 75])
        # plt.ylim([-math.pi, math.pi])

        # for t, _ in enumerate(encoder_times):
        #     u_t = np.array([[u_l[t]], [u_r[t]]], dtype = np.float64)
        #     x_t_est = propagate_state(x_t_prev, u_t)
        #     state_estimates_encoder[:, t] = x_t_est.reshape((N,))
        #
        #     x_t_prev = x_t_est

        # plt.figure(2)
        # plt.scatter(state_estimates[0,:], state_estimates[1,:])
        # plt.show()
        # plt.figure(3)
        # plt.scatter(encoder_times, state_estimates[2,:])
        # plt.show()

        NUM_BEAMS = int(263/3)
        for t, _ in enumerate(encoder_times):

            print(encoder_times[t])

            # Get control input
            u_t = np.array([[u_l[t]], [u_r[t]]], dtype = np.float64)
            # print("u_t: ", u_t.shape)

            # figure out if we will correct
            # while (z_range_var[lidar_index] >= 1):
            #     lidar_index += 1

            if (lidar_index < len(lidar_times)-NUM_BEAMS and lidar_times[lidar_index] > encoder_times[t] and lidar_times[lidar_index] < encoder_times[t+1]+0.9):

                ax.imshow(im2, extent=[low_x, high_x, high_y, low_y ])
                plt.xlim([x_t_prev[0]-100,x_t_prev[0]+100])
                plt.ylim([high_y, low_y])

                print("Predicting and correcting")
                x_t_pred, sigma_t_pred = prediction_step(x_t_prev, sigma_t_prev, u_t)
                # x_t_pred[2] = 0
                # Get measurement

                z_t = np.zeros((NUM_BEAMS, 1), dtype = np.float64)
                azimuth_t = np.zeros((NUM_BEAMS, 1), dtype = np.float64)
                Q_t = np.zeros((NUM_BEAMS, NUM_BEAMS), dtype = np.float64)

                for i in range(NUM_BEAMS):
                    z_t[i] = z_range_mu[lidar_index+i]
                    azimuth_t[i] = z_azimuth_mu[lidar_index+i]
                    Q_t[i,i] = z_range_var[lidar_index+i]
                    #draw_points(z_t[i], azimuth_t[i], x_t_pred[0], x_t_pred[1], x_t_pred[2])

                # print("z_t: ", z_t.shape)
                #if (abs(np.mean(azimuth_t)) < 3*math.pi/4 and abs(np.mean(azimuth_t)) > math.pi/4):
                x_t_est, sigma_t_est = correction_step(x_t_pred, sigma_t_pred, z_t, azimuth_t, Q_t, x_t_prev)
                # else:
                #     x_t_est = x_t_pred
                #     sigma_t_est = R_t


                lidar_index += NUM_BEAMS

                # x_t_est = propagate_state(x_t_prev, u_t)
                # sigma_t_est = R_t
                # x_t_est[2] = 0
                #plt.subplot(2,1,1)
                plt.scatter(x_t_est[0], x_t_est[1], s=5, marker='.', color='g')
                plt.scatter(x_truth, y_truth, s= 5, marker = 'x', color='r')

                # plt.subplot(2,1,2)
                # plt.scatter(encoder_times[t], x_t_est[2], color='b')
                # plt.scatter(encoder_times[t], x_t_pred[2], color='r')
                # plt.xlim([55, 75])
                # plt.ylim([-math.pi, math.pi])

                plt.pause(0.0001)
                ax.clear()

            else:
                # we can't predict without correcting
                print("Propagating only")
                # G_x_t = calc_prop_jacobian_x(x_t_prev, u_t)
                # G_u_t = calc_prop_jacobian_u(x_t_prev, u_t)
                # x_t_pred = propagate_state(x_t_prev, u_t)
                # x_t_est = x_t_pred
                #
                # print(G_x_t)
                # sigma_t_est= sigma_t_prev + R_t #+ G_u_t.dot(R_t).dot(np.transpose(G_u_t))  G_x_t.dot(sigma_t_prev).dot(np.transpose(G_x_t)) +
                x_t_est, sigma_t_est = prediction_step(x_t_prev, sigma_t_prev, u_t)
                # if encoder_times[t] > 60:
                #     ax.imshow(im2, extent=[low_x, high_x, high_y, low_y ])
                #     plt.xlim([low_x,int(high_x/5)])
                #     plt.ylim([high_y, low_y])
                #     plt.scatter(x_t_est[0], x_t_est[1], s=2, marker='.', color='b')
                #     plt.pause(0.0001)
                #sigma_t_est = R_t
                # x_t_est[2] = 0

                # x_t_est = propagate_state(x_t_prev, u_t)


            x_estimates.append(x_t_est[0])
            y_estimates.append(x_t_est[1])
            theta_estimates.append(x_t_est[2])

            x_t_prev = x_t_est
            sigma_t_prev = sigma_t_est


            print("Estimated state: \n", x_t_est)
            print("Estimated sigma: \n", sigma_t_est)



    plt.figure(1)
    fig, ax = plt.subplots()
    low_x, low_y = grid_to_xy(0,0)
    high_x, high_y = grid_to_xy(ROWS-1,COLS-1)
    ax.imshow(im2, extent=[low_x, high_x, high_y, low_y ])
    #plt.xlim([int(high_x/4),int(3*high_x/4)])
    #plt.ylim([high_y, low_y])
    ax.scatter(x_estimates, y_estimates, s=2, marker='.')
    ax.scatter(x_estimates, y_estimates, s=2, marker='.',color='r')

    plt.show()

    plt.figure(2)
    plt.scatter(encoder_times, x_estimates)
    plt.scatter(encoder_times, u_l, c="r")
    plt.scatter(encoder_times, u_r, c="k")
    plt.show()

    plt.figure(3)
    plt.scatter(encoder_times, y_estimates)
    plt.show()

    plt.figure(4)
    plt.scatter(encoder_times, theta_estimates)
    plt.show()

    im = Image.open('map.png')
    #im = Image.open('map_pic.jpg')
    pixelMap = im.load()
    #
    # for i in range(len(state_estimates[0,:])):
    #     x = state_estimates[0,i]
    #     y = state_estimates[1,i]
    #     plot_pixel(pixelMap,x,y,(0,200,200,255))
    #
    #
    for i in range(len(x_vals)):
        plot_pixel(pixelMap,x_vals[i],y_vals[i],(0,200,0,255))

    with open('ukf.csv', 'w', newline='') as csvfile:
        fieldnames = ['times', 'xd', 'x', 'yd', 'y', 'theta', 'w']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for t, _ in enumerate(time_stamps):
            writer.writerow({'times': encoder_times[t], 'x_sub': state_estimates[0][t], \
             'y_sub': state_estimates[1][t], 'theta_sub': state_estimates[2][t], 'x_simp': state_estimates_simple[0][t], \
              'y_simp': state_estimates_simple[1][t], 'theta_simp': state_estimates_simple[2][t], "correcting": correcting[t]})



    im.show()
    im.save("out.png")
    im.close()



if __name__ == "__main__":
    main()
