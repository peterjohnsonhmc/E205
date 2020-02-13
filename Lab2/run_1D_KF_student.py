"""
Peter Johnson and Pinky King based on code by
Author: Andrew Q. Pham
Email: apham@g.hmc.edu
Date of Creation: 2/8/20
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
            data[h].append(float(element))

    f.close()

    yaw_data = data["Yaw(degrees)"]

    return yaw_data


def prediction_step(x_t_prev, sigma_sq_t_prev):
    """Compute the prediction of 1D Kalman Filter

    Parameters:
    x_t_prev        -- the previous state estimate
    sigma_sq_t_prev -- the previous variance estimate

    Returns:
    x_bar_t         -- the predicted state estimate of time t
    sigma_sq_bar_t  -- the predicted variance estimate of time t
    """

    """STUDENT CODE START"""
    x_bar_t = x_t_prev
    sigma_sq_bar_t = sigma_sq_t_prev
    """STUDENT CODE END"""

    return [x_bar_t, sigma_sq_bar_t]


def correction_step(x_bar_t, z_t, sigma_sq_bar_t, sigma_sq_z):
    """Compute the correction of 1D Kalman Filter

    Parameters:
    x_bar_t         -- the predicted state estimate of time t
    z_t             -- the measured state of time t
    sigma_sq_bar_t  -- the predicted variance of time t
    sigma_z         -- the variance of sensor measurement


    Returns:
    x_est_t         -- the filtered state estimate of time t
    sigma_sq_est_t  -- the filtered variance estimate of time t
    """

    Kt = sigma_sq_bar_t/(sigma_sq_bar_t+sigma_sq_z)
    sigma_sq_est_t = sigma_sq_bar_t - Kt*sigma_sq_bar_t
    x_est_t = x_bar_t + Kt*(z_t-x_bar_t)

    return [x_est_t, sigma_sq_est_t]


def wrap_to_360(angle):
    """Wrap angle data to [0, 360]"""
    return (angle + 360) % 360


def plot_yaw(yaw_dict, time_stamps, title=None, xlim=None, ylim=None):
    """Plot yaw data"""
    plt.plot(np.asarray(time_stamps),
             np.array(yaw_dict["measurements"]),
             '.',
             markersize=1)
    plt.plot(np.asarray(time_stamps),
             np.array(yaw_dict["estimates"]),
             '.',
             markersize=1)
    plt.plot(np.asarray(time_stamps),
             np.asarray(yaw_dict["plus_2_stddev"]),
             '.-',
             markersize=1,
             linewidth=0.5)
    plt.plot(np.asarray(time_stamps),
             np.asarray(yaw_dict["minus_2_stddev"]),
             '.-',
             markersize=1,
             linewidth=0.5)
    plt.legend(["Raw Data", "Estimate", "+2$\sigma$", "-2$\sigma$"])
    plt.title(title)
    plt.ylabel("Yaw (Degrees)")
    plt.xlabel("Time (s)")
    plt.xlim(xlim)
    plt.ylim(ylim)


def main():
    """Run a 1D Kalman Filter on logged yaw data from a BNO055 IMU."""

    #filepath = ".\"
    filename = "2020-02-08_08_52_01.csv"
    #yaw_data = load_data(filepath + filename)
    yaw_data = load_data(filename)

    """STUDENT CODE START"""
    SENSOR_MODEL_VARIANCE = 1.9273
    """STUDENT CODE END"""

    #  Initialize filter
    yaw_est_t_prev = yaw_data[0]
    var_t_prev = SENSOR_MODEL_VARIANCE
    yaw_dict= {}
    yaw_dict["measurements"] = yaw_data
    yaw_dict["estimates"] = []
    yaw_dict["plus_2_stddev"] = []
    yaw_dict["minus_2_stddev"] = []
    time_stamps = []

    #  Run filter over data
    for t, _ in enumerate(yaw_data):
        yaw_pred_t, var_pred_t = prediction_step(yaw_est_t_prev, var_t_prev)

        # To be explicit for teaching purposes, we are getting
        # the measurement with index 't' to show how we get a
        # new measurement each time step. To be more pythonic we could
        # replace the '_' above with 'yaw_meas'
        yaw_meas = yaw_data[t]
        var_z = SENSOR_MODEL_VARIANCE

        yaw_est_t, var_est_t = correction_step(yaw_pred_t,
                                               yaw_meas,
                                               var_pred_t,
                                               var_z)

        #  Format the printouts
        #sys.stdout.write("Yaw State Estimate: {0}\n\
        #                 \r Yaw Raw Data:       {1}\
        #                 \033[A\r".format(yaw_est_t, yaw_meas))
        #sys.stdout.flush()

        sys.stdout.write("\r Yaw State Estimate: %f    Yaw Measured: %f \n" % (yaw_est_t,yaw_meas))
        sys.stdout.flush()


        sys.stdout.write("Estimated variance: {0}\n\r".format(var_est_t))
        sys.stdout.flush()

        #  Pause the printouts to simulate the real data rate
        dt = 1/13.  # seconds
        time_stamps.append(dt*t)

        #  Comment out sleep to visualize the plot immediately
        # time.sleep(dt)

        #  For clarity sake/teaching purposes, we explicitly update t->(t-1)
        yaw_est_t_prev = yaw_est_t
        var_est_t_prev = var_est_t

        # Pack data away into yaw dictionary for plotting purpose
        plus_2_stddev = wrap_to_360(yaw_est_t + 2*math.sqrt(var_est_t))
        minus_2_stddev = wrap_to_360(yaw_est_t - 2*math.sqrt(var_est_t))

        yaw_dict["estimates"].append(yaw_est_t)
        yaw_dict["plus_2_stddev"].append(plus_2_stddev)
        yaw_dict["minus_2_stddev"].append(minus_2_stddev)

    print("\n\nDone filtering...plotting...")

    # Plot raw data and estimate
    plt.figure(1)
    plt.suptitle("1D Kalman Filtering: Yaw Measurements")
    plt.subplot(1, 2, 1)
    plot_yaw(yaw_dict, time_stamps, title="Full Log")
    plt.subplot(1, 2, 2)
    plot_yaw(yaw_dict,
             time_stamps,
             title="Zoomed",
             xlim=[14, 24],
             ylim=[280, 345])
    plt.show()

    print("Exiting...")

    return 0


if __name__ == "__main__":
    main()
