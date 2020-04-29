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
alpha = .01
beta = alpha**2-1 # assume Gaussian
k = 0
lamda = alpha**2*(N-k)
#lamda = -2
PRINTING = False
PIX_2_M = 7.605 #7.82
ROWS = 1395 #1492
COLS = 5329 #5352
ROW_OFFSET_A = 785 # 835
COL_OFFSET_A = 745 # 760
R_t = np.array([[0.00435**2,0,0], [0,0.00435**2,0], [0,0,(0.0000435/L)**2]])
global MAP
global plottingMAP
global im2

#NUmber of beams per time step
NUM_BEAMS = 7

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
    print("Sigma point: ", x_t_pred)
    x, y, theta = x_t_pred
    z_t_pred = np.zeros((NUM_BEAMS, 1))

    for i in range(NUM_BEAMS):

        azimuth = azimuth_t[i]
        # convert from lidar frame to global frame
        azimuth = wrap_to_pi(theta + azimuth)
        print("Azimuth :", azimuth)
        m = math.tan(azimuth)
        x_curr = x
        y_curr = y
        row_curr, col_curr = xy_to_grid(x, y)
        if (abs(m) > 1):
            m = 1/m # in the case that we flip x and y
            flipped_azimuth = -(azimuth-math.pi/2)
            # because of how y is indexed, we want to decrease the col to move up
            # and increase the col to move down
            if (flipped_azimuth > math.pi/2 or flipped_azimuth < -math.pi/2):
                increment = 1
            else:
                increment = -1
            while (sum(MAP[col_curr,row_curr]) < 300):
                row_curr = row_curr+increment
                x_curr, y_curr = grid_to_xy(row_curr, col_curr)
                x_curr = m*y_curr + x
                _, col_curr = xy_to_grid(x_curr, y_curr)
                plot_1_pixel(plottingMAP, x_curr, y_curr, (200,0,0,255))
        else:
            if (azimuth > math.pi/2 or azimuth < -math.pi/2):
                increment = -1
            else:
                increment = 1
            while (sum(MAP[col_curr,row_curr]) < 300):
                col_curr = col_curr+increment
                x_curr, y_curr = grid_to_xy(row_curr, col_curr)
                #print(x_curr,y_curr)
                y_curr = m*x_curr + y
                row_curr, _ = xy_to_grid(x_curr, y_curr)
                plot_1_pixel(plottingMAP, x_curr, y_curr, (200,0,0,255))
                #print(col_curr, row_curr)

        z_t_pred[i] = distance(x, y, x_curr, y_curr)
    print("range_pred: ", z_t_pred)
    return z_t_pred

def in_tunnel(sigma_point, mu):
    # print("in in_tunnel function")
    x, y, theta = sigma_point
    x_m, y_m, theta_m = mu
    row, col = xy_to_grid(x, y)
    if (sum(MAP[col,row]) < 300):
        return sigma_point
    else:
        # draw a line from sigma_point to mu, find the first black point on that line
        delta_x = x_m-x
        delta_y = y_m-y
        if (delta_x == 0):
            m = 0
            col_curr, row_curr = col, row
            x_curr, y_curr = x, y
            while (in_bounds(row_curr, col_curr)==False or sum(MAP[col_curr,row_curr]) > 300):
                row_curr = row_curr+1
                x_curr, y_curr = grid_to_xy(row_curr, col_curr)
                x_curr = m*y_curr + x
                _, col_curr = xy_to_grid(x_curr, y_curr)
        else:
            m = delta_y/delta_x
            col_curr, row_curr = col, row
            x_curr, y_curr = x, y
            while (in_bounds(row_curr, col_curr)==False or sum(MAP[col_curr,row_curr]) > 300):
                col_curr = col_curr+1
                x_curr, y_curr = grid_to_xy(row_curr, col_curr)
                y_curr = m*x_curr + y
                row_curr, _ = xy_to_grid(x_curr, y_curr)

        sigma_point[0] = x_curr
        sigma_point[1] = y_curr
        return sigma_point

def in_bounds(row, col):
    if (row < ROWS and row >= 0 and col < COLS and col >= 0):
        return True
    else:
        return False

def get_chi(x, sigma):

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

    return chi

def prediction_step(x_t_prev, sigma_t_prev, u_t):

    print("u_t: ", u_t)
    print("x_t_prev: ", x_t_prev)

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
    w_c_i = 1/(2*(N+lamda))
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


def correction_step(x_t_pred, sigma_t_pred, z_t, azimuth_t, Q_t):

    #print("range: ", z_t[0])
    #z_t = z_t.reshape((NUM_BEAMS,1))

    # get the predicted chi from the predicted sigma and x
    chi_t_pred = get_chi(x_t_pred, sigma_t_pred)
    #print("chi_t_pred: ", chi_t_pred)


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
    print("Residual: ", residual)
    percent_off = 100*np.min(residual)/np.mean(z_t)
    print("percent_off: ", percent_off, "%")
    print("Residual: ", z_t - z_t_pred)
    if abs(percent_off) > 30:
        x_t_est = x_t_pred
    else:
        x_t_est = x_t_pred.reshape((N,1)) + K_t.dot(residual)
    # Update sigma_t_est
    #print("delta_sigma:", K_t.dot(S_t).dot(np.transpose(K_t)))
    sigma_t_est = sigma_t_pred - K_t.dot(S_t).dot(np.transpose(K_t))
    for i in range(N):
        for j in range(N):
            if sigma_t_est[i,j] < 0:
                sigma_t_est[i,j] = 1e-10

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

def main():
    """Run a EKF on logged data from IMU and LiDAR moving in a box formation around a landmark"""

    #np.random.seed(28)

    im = Image.open('map.png')
    global MAP
    MAP = im.load()

    global im2
    im2 = Image.open('map.png')
    global plottingMAP
    plottingMAP = im2.load()

    filepath = "filtered_data/01M/"
    filename =  "01M_encoder_filtered" #"2020_2_26__17_21_59"
    encoder_data = load_data(filepath + filename)

    filename =  "01M_lidar_multi_" + str(NUM_BEAMS) #"2020_2_26__17_21_59"
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

    plt.figure(0)
    # plt.subplot(2,1,1)
    fig, ax = plt.subplots()
    low_x, low_y = grid_to_xy(0,0)
    high_x, high_y = grid_to_xy(ROWS-1,COLS-1)
    ax.imshow(im2, extent=[low_x, high_x, high_y, low_y ])
    plt.xlim([low_x,int(high_x/5)])
    plt.ylim([high_y, low_y])

    # plt.subplot(2,1,2)
    # plt.xlim([55, 75])
    # plt.ylim([-math.pi, math.pi])




    for t, _ in enumerate(encoder_times):

        print(encoder_times[t])

        # Get control input
        u_t = np.array([[u_l[t]], [u_r[t]]], dtype = np.float64)
        # print("u_t: ", u_t.shape)

        # figure out if we will correct
        # while (z_range_var[lidar_index] >= 1):
        #     lidar_index += 1
        if ( lidar_times[lidar_index] > encoder_times[t] and lidar_times[lidar_index] < encoder_times[t+1]+0.9):
            print("Predicting and correcting")
            x_t_pred, sigma_t_pred = prediction_step(x_t_prev, sigma_t_prev, u_t)
            # Get measurement
            z_t = np.zeros((NUM_BEAMS, 1), dtype = np.float64)
            azimuth_t = np.zeros((NUM_BEAMS, 1), dtype = np.float64)
            Q_t = np.zeros((NUM_BEAMS, NUM_BEAMS), dtype = np.float64)

            for i in range(NUM_BEAMS):
                z_t[i] = z_range_mu[lidar_index+i]
                azimuth_t[i] = z_azimuth_mu[lidar_index+i]
                Q_t[i,i] = z_range_var[lidar_index+i]

            # print("z_t: ", z_t.shape)
            x_t_est, sigma_t_est = correction_step(x_t_pred, sigma_t_pred, z_t, azimuth_t, Q_t)

            lidar_index += NUM_BEAMS

            #plt.subplot(2,1,1)
            plt.scatter(x_t_pred[0], x_t_pred[1], s=5, marker='.', color='r')
            plt.scatter(x_t_est[0], x_t_est[1], s=5, marker='.', color='b')
            plt.xlim([low_x,int(high_x/5)])
            plt.ylim([high_y, low_y])

            # plt.subplot(2,1,2)
            # plt.scatter(encoder_times[t], x_t_est[2], color='b')
            # plt.scatter(encoder_times[t], x_t_pred[2], color='r')
            # plt.xlim([55, 75])
            # plt.ylim([-math.pi, math.pi])
            #
            plt.pause(0.0001)


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
            # plt.scatter(x_t_est[0], x_t_est[1], s=10, marker='.', color='b')
            x_t_est, sigma_t_est = prediction_step(x_t_prev, sigma_t_prev, u_t)
            sigma_t_est = R_t


        state_estimates[:, t] = x_t_est.reshape((N,))
        x_t_prev = x_t_est
        sigma_t_prev = sigma_t_est


        print("Estimated state: \n", x_t_est)
        print("Estimated sigma: \n", sigma_t_est)



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



    im.show()
    im.save("out.png")
    im.close()



if __name__ == "__main__":
    main()
