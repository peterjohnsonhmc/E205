import csv
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
from statistics import stdev
import pandas as pd

# constant values
alpha = 1
beta = .315
gamma = .5
dt = 0.1

# based on starter code from a previous lab
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
    data = {}
    header = ["times", "u_vert", "u_left", "u_right", "x", "y", "z", "compass"]
    for h in header:
        data[h] = []

    for row in file_reader:
        for h, element in zip(header, row):
            data[h].append(float(element))

    # Calculate v, w, and s from the control inputs
    # Our input vector will be u = [v s w]T, which allows for simpler Jacobians
    data["v"] = []
    data["w"] = []
    data["s"] = []
    for i in range(0, len(data["times"])):
        u_v = data["u_vert"][i]
        u_l = data["u_left"][i]
        u_r = data["u_right"][i]
        # Calculate v, w, and s from given equations and add to data dictionary
        data["v"].append(float(alpha*(u_l+u_r)))
        data["w"].append(float(beta*(-u_l+u_r)))
        data["s"].append(float(gamma*u_v))

    f.close()

    return data

# based on starter code from a previous lab
def wrap_to_pi(angle):
    """Wrap angle data in radians to [-pi, pi]

    Parameters:
    angle (np.double)   -- unwrapped angle

    Returns:
    angle (np.double)   -- wrapped angle
    """
    while angle >= math.pi:
        angle -= 2*math.pi

    while angle <= -math.pi:
        angle += 2*math.pi
    return angle


def propagate_state(x_prev_t, u_t):
    """Predict the current state based on the previous state and control input

    Parameters:
    x_prev_t (np.array) -- the previous state estimate
    u_t (np.array)      -- the current input

    Returns:
    x_pred_t (np.array) -- the current state prediction
    """
    x, y, z, theta = x_prev_t
    v, s, w = u_t
    x_pred_t = np.array([[x + v*dt*math.cos(theta)],
                         [y + v*dt*math.sin(theta)],
                         [z + s*dt],
                         [wrap_to_pi(theta + w*dt)]],
                        dtype=float)
    return x_pred_t


def find_Gx(x_prev_t, u_t):
    """Calculate the Jacobian of the state transition function
    with respect to state variables

    Parameters:
    x_prev_t (np.array) -- the previous state estimate
    u_t (np.array)      -- the current input

    Returns:
    Gx (np.array)       -- the Jacobian matrix Gx
    """
    _, _, _, theta = x_prev_t
    v, _, _ = u_t
    Gx = np.array([[1, 0, 0, -v*dt*math.sin(theta)],
                   [0, 1, 0,  v*dt*math.cos(theta)],
                   [0, 0, 1,                     0],
                   [0, 0, 0,                     1]],
                  dtype=float)
    return Gx


def find_Gu(x_prev_t):
    """Calculate the Jacobian of the state transition function
    with respect to input variables

    Parameters:
    x_prev_t (np.array) -- the previous state estimate

    Returns:
    Gu (np.array)       -- the Jacobian matrix Gu
    """
    _, _, _, theta = x_prev_t
    Gu = np.array([[dt*math.cos(theta), 0, 0],
                   [dt*math.sin(theta), 0, 0],
                   [0, dt, 0],
                   [0, 0, dt]],
                  dtype=float)
    return Gu


def prediction_step(x_prev_t, sigma_prev_t, u_t, R_t):
    """Predict the current state and associated covariance matrix

    Parameters:
    x_prev_t (np.array)     -- the previous state estimate
    sigma_prev_t (np.array) -- the previous covariance estimate
    u_t (np.array)          -- the current input
    R_t (np.array)          -- covariance matrix for odometry

    Returns:
    x_pred_t (np.array)     -- the predicted state
    sigma_pred_t (np.array) -- the predicted covariance
    """
    # predict the state using the state transition function g(x,u)
    x_pred_t = propagate_state(x_prev_t, u_t)
    x_pred_t = x_pred_t.reshape((4,))

    # get Jacobian matrices and their transposes
    Gx_t = find_Gx(x_prev_t, u_t)
    Gx_t_T = np.transpose(Gx_t)
    Gu_t = find_Gu(x_prev_t)
    Gu_t_T = np.transpose(Gu_t)

    # calculate the predicted variance
    sigma_pred_t = Gx_t.dot(sigma_prev_t).dot(Gx_t_T) + Gu_t.dot(R_t).dot(Gu_t_T)

    return [x_pred_t, sigma_pred_t]


def kalman_gain(sigma_pred_t, Q_t):
    """Find the Kalman gain matrix

    Parameters:
    sigma_pred_t (np.array) -- the predicted covariance
    Q_t (np.array)          -- covariance matrix for measurement

    Returns:
    K_t (np.array)          -- the Kalman gain matrix
    """
    # H_t = I, so no need to include it in calculations
    K_t = sigma_pred_t.dot(np.linalg.inv(sigma_pred_t + Q_t))
    return K_t


def correction_step(x_pred_t, sigma_pred_t, z_t, Q_t):
    """Correct the state and covariance predictions to get the estimated state
    and its covariance matrix

    Parameters:
    x_pred_t (np.array)     -- the predicted state
    sigma_pred_t (np.array) -- the predicted covariance
    z_t (np.array)          -- the currrent measurement
    Q_t (np.array)          -- covariance matrix for measurement

    Returns:
    x_est_t (np.array)     -- the estimated state
    sigma_est_t (np.array) -- the estimated covariance
    """
    K_t = kalman_gain(sigma_pred_t, Q_t)

    z_pred_t = x_pred_t # very simple h function
    resid = z_t - z_pred_t
    resid[3] = wrap_to_pi(resid[3]) # angle wrapping when adding angles

    x_est_t = x_pred_t + K_t.dot(resid)
    x_est_t = x_est_t.reshape((4,))
    sigma_est_t = (np.eye(4, dtype = float) - K_t).dot(sigma_pred_t) # H_t = I

    return [x_est_t, sigma_est_t]


def find_sigma(data_set):
    """Finds the standard deviation, sigma, of a 1D data_set

    Parameters:
    data_set (list)    -- data set to find the sigma of

    Returns:
    sigma (float)      -- the standard deviation of data_set
    """
    mu = np.mean(data_set)
    sigma = stdev(data_set, mu)
    return sigma


def main():
    # read in data and save the useful data in lists
    filename = "E205_Exam_2020.csv" # "data.csv" # this is what I ran but changing name for Prof Clark
    data = load_data(filename)
    times = data["times"]
    x_gps = data["x"]
    y_gps = data["y"]
    z_alt = data["z"]
    yaw_c = data["compass"]
    v     = data["v"]
    s     = data["s"]
    w     = data["w"]

    # variances for the measurements
    # the blimp is not moving for time steps 500 - 800, so this is a
    # good section to find the standard deviation for
    # ASSUMPTION: all covariances NOT on the diagonal are 0 (all measurements
    # are independent)
    dataframe = [x_gps[500:800], y_gps[500:800], z_alt[500:800], yaw_c[500:800]]

    Q_t = np.array([[find_sigma(x_gps[500:800])**2, 0, 0, 0],
                    [0, find_sigma(y_gps[500:800])**2, 0, 0],
                    [0, 0, find_sigma(z_alt[500:800])**2, 0],
                    [0, 0, 0, find_sigma(yaw_c[500:800])**2]],
                    dtype = float)

    # variances for the odometry
    # the odometry is 0 for time steps 500 - 800, so this is a
    # good section to find the standard deviation for
    # ASSUMPTION: all covariances NOT on the diagonal are 0 (all odometry
    # variables are independent)
    R_t = np.array([[find_sigma(v[500:800])**2, 0, 0],
                    [0, find_sigma(s[500:800])**2, 0],
                    [0, 0, find_sigma(w[500:800])**2]],
                    dtype = float)

    # starting state is all 0s
    x_prev_t = np.array([[0], [0], [0], [0]], dtype = float)
    x_prev_t = x_prev_t.reshape((4,))
    # starting covariance matrix is the covariance of our measurement
    sigma_prev_t = Q_t
    # set up matrices to store data
    state_estimates = np.empty((4, len(times)))
    sigma_estimates = np.empty((4, 4, len(times)))

    # loop through all the time steps and filter
    for t, _ in enumerate(times):

        # extract odometry and measurement from data
        u_t = np.array([[v[t]], [s[t]], [w[t]]], dtype = float)
        u_t.reshape((3,))
        z_t = np.array([[x_gps[t]], [y_gps[t]], [z_alt[t]], [yaw_c[t]]], dtype = float)
        z_t = z_t.reshape((4,))

        # prediction and correction
        x_pred_t, sigma_pred_t = prediction_step(x_prev_t, sigma_prev_t, u_t, R_t)
        x_est_t, sigma_est_t   = correction_step(x_pred_t, sigma_pred_t, z_t, Q_t)

        # save estimates to plot later
        state_estimates[:,t] = x_est_t
        sigma_estimates[:,:,t] = sigma_est_t

        # the current estimates become the previous estimates as we move to
        # next time step
        x_prev_t = x_est_t
        sigma_prev_t = sigma_est_t

    # plot theta within 2sigma and sigma bounds
    plt.figure(1)
    stdev_theta = []
    for var in (sigma_estimates[3][3][:]):
        stdev_theta.append(math.sqrt(var)) # standard deviation is sqrt of variance
    plus_2_stddev = list(map(lambda x,y:x+2*y, state_estimates[3][:], stdev_theta))
    minus_2_stddev = list(map(lambda x,y:x-2*y, state_estimates[3][:], stdev_theta))
    # plot zoomed out and zoomed in
    for i in range(1,3):
        plt.subplot(1,2,i)
        plt.plot(times, state_estimates[3][:], marker='.', c='k')
        plt.plot(times, plus_2_stddev, c='g', alpha=.8)
        plt.plot(times, minus_2_stddev, c='r', alpha=0.8)
        plt.scatter(times, yaw_c, c="b", alpha=0.5, marker='.', edgecolors='none')
        plt.ylabel("\u03b8 (rad)")
        plt.xlabel("Time (s)")
        plt.legend(["Estimated \u03b8", "Est. \u03b8 + 2\u03C3", "Est. \u03b8 - 2\u03C3", "Measured \u03b8"])
        if (i == 2):
            plt.xlim([-.1,10.5]) # uncomment these lines to zoom in
            plt.ylim([-.2,.05])
    plt.suptitle("Estimated \u03b8 and Standard Deviation Over Time")
    plt.show()

    # plot 3D estimated and measured positions
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(state_estimates[0][:], state_estimates[1][:], state_estimates[2][:], marker='.', c='k')
    ax.scatter(x_gps, y_gps, z_alt, c="b", alpha=0.3, marker='.')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Plot of Estimated and Measured Positions")
    ax.legend(["Estimated Position", "Measured Position"], ncol=2, framealpha=1, bbox_to_anchor=(.9, -.02))
    plt.show()

    return 0

if __name__ == "__main__":
    main()
