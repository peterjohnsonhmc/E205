"""
Peter Johnson and Pinky King based on code by
Author: Andrew Q. Pham
Email: apham@g.hmc.edu
Date of Creation: 4/1/20
Description:
    1D Kalman Filter implementation to filter logged yaw data from a BNO055 IMU
    This code is for teaching purposes for HMC ENGR205 System Simulation Lab 2
"""

import csv
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
from statistics import stdev
from matplotlib.patches import Ellipse


alpha = 1
beta = 0.315
dt = 0.1

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
    header = ["Time", "u_l", "u_r", "gpsx", "gpsy", "angle"]
    for h in header:
        data[h] = []

    for row in file_reader:
        for h, element in zip(header, row):
            data[h].append(float(element))

    data["v"] = []
    data["w"] = []
    for i in range(0,len(data["u_l"])):
        u_l = data["u_l"][i]
        u_r = data["u_r"][i]
        data["v"].append(float(alpha*(u_l+u_r)))
        data["w"].append(float(beta*(-u_l+u_r)))

    f.close()

    return data

def wrap_to_pi(angle):
    while angle >= math.pi:
        angle -= 2*math.pi

    while angle <= -math.pi:
        angle += 2*math.pi
    return angle

def propogate_state(x_prev_t, u_t):
    x, y, theta = x_prev_t
    v, w = u_t
    x_pred_t = np.array([x + v*dt*math.cos(theta),
                         y + v*dt*math.sin(theta),
                         wrap_to_pi(theta + dt*w)],
                         dtype = float)
    return x_pred_t

def calc_jac_Gx(x_prev_t, u_t):
    x, y, theta = x_prev_t
    v, w = u_t
    Gx = np.array([[1, 0, -v*dt*math.sin(theta)],
                   [0, 1,  v*dt*math.cos(theta)],
                   [0, 0,                     1]],
                    dtype = float)
    return Gx

def calc_jac_Gu(x_prev_t, u_t):
    x, y, theta = x_prev_t
    v, w = u_t
    Gu = np.array([[dt*math.cos(theta), 0],
                   [dt*math.sin(theta), 0],
                   [0,                 dt]],
                   dtype = float)
    return Gu

def prediction_step(x_prev_t, sigma_prev_t, u_t):
    x_pred_t = propogate_state(x_prev_t, u_t)
    x_pred_t = x_pred_t.reshape((3,))

    R_t = np.array([[.077**2, 0],
                    [0, .024**2]], dtype=float)
    G_x_t = calc_jac_Gx(x_prev_t, u_t)
    G_u_t = calc_jac_Gu(x_prev_t, u_t)
    sigma_pred_t = G_x_t.dot(sigma_prev_t).dot(np.transpose(G_x_t)) + G_u_t.dot(R_t).dot(np.transpose(G_u_t))
    return [x_pred_t, sigma_pred_t]

def kalman_gain(H_t, sigma_pred_t):
    Q_t = np.array([[.55**2,0,0],
                    [0,.5**2,0],
                    [0,0,.01**2]], dtype = float)
    H_t_T = np.transpose(H_t)
    K = sigma_pred_t.dot(H_t_T).dot(np.linalg.inv(H_t.dot(sigma_pred_t).dot(H_t_T) + Q_t))
    return K

def correction_step(x_pred_t, sigma_pred_t, z_t):
    H_t = np.eye(3, dtype=float)
    K_t = kalman_gain(sigma_pred_t, H_t)

    z_bar_t = x_pred_t
    resid = z_t - z_bar_t
    resid[2] = wrap_to_pi(resid[2])
    x_est_t = x_pred_t + K_t.dot(resid)
    x_est_t = x_est_t.reshape((3,))
    sigma_est_t = (np.eye(3, dtype=float)-K_t.dot(H_t)).dot(sigma_pred_t)

    return [x_est_t, sigma_est_t]

def find_mu_sigma(data_set):
    mu = np.mean(data_set)
    sigma = stdev(data_set, mu)
    return [mu, sigma]

def main():
    filename = "measurements.csv"
    data = load_data(filename)

    times = data["Time"]
    gpsx = data["gpsx"]
    gpsy = data["gpsy"]
    v = data["v"]
    w = data["w"]
    compass = data["angle"]

    x_prev_t = np.array([0,0,0])
    sigma_prev_t = np.identity(3)

    state_estimates = np.empty((3, len(times)))
    covariance_estimates = np.empty((3, 3, len(times)))

    [mu, sigma] = find_mu_sigma(w[1:200])
    print(mu)
    print(sigma)

    for t, _ in enumerate(times):

        u_t = np.array([[v[t]], [w[t]]], dtype = float)
        z_t = np.array([gpsx[t], gpsy[t], compass[t]], dtype = float)

        x_pred_t, sigma_pred_t = prediction_step(x_prev_t, sigma_prev_t, u_t)
        x_est_t, sigma_est_t = correction_step(x_pred_t, sigma_pred_t, z_t)

        state_estimates[:, t] = x_est_t
        covariance_estimates[:, :, t] = sigma_est_t

        x_prev_t = x_est_t
        sigma_prev_t = sigma_est_t


    # Plot raw data and estimate
    plt.figure(1)
    plt.suptitle("EKF Localization: X & Y Measurements")
    plt.scatter(state_estimates[0][:], state_estimates[1][:])
    plt.scatter(gpsx, gpsy, alpha=0.5)
    # plt.xlabel('X [m]')
    # plt.ylabel('Y [m]')
    # plt.xlim([-10, 20])
    # plt.ylim([-20, 10])
    plt.show()

    plt.figure(2)
    plt.scatter(times, v)
    plt.show()

    plt.figure(3)
    plt.scatter(times, w)
    plt.show()

    plt.figure(4)
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    for t in range(len(times)):
        x = state_estimates[0][t]
        y = state_estimates[1][t]
        xg = gpsx[t]
        yg = gpsy[t]

        if (t % 10 == 0):
            xvar = covariance_estimates[0][0][t]
            yvar = covariance_estimates[1][1][t]
            theta = state_estimates[2][t]
            el = Ellipse(xy=(x,y),
                         width=100*yvar, height=100*xvar,
                         angle=theta)
            ax.add_artist(el)
            el.set_clip_box(ax.bbox)
            el.set_alpha(0.3)
            el.set_facecolor('g')
        ax.scatter(x, y, c='r', marker='.')
        ax.scatter(xg, yg, c='b', marker='.')
        #ax.pause(0.0005)
    ax.set_ylabel("Y position (Global Frame, m)")
    ax.set_xlabel("X position (Global Frame, m)")
    #ax.legend(["Expected Path", "Estimated Position", "GPS Position"], loc='center right')
    plt.show()

    return 0


if __name__ == "__main__":
    main()
