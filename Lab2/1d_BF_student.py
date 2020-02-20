"""
Peter Johnson and Pinky King based on code by
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
from scipy.stats import norm, halfnorm, uniform
import scipy.stats

#Global Variables
#Probability density parameters
STOP_MU = 0
STOP_STD = 0
MOVE_MU = 0
MOVE_STD = 0

#Conditional Probabilities
#p(X_t = Stopped | x_t_p = Stopped)
pS_S = 0.6
pS_M = 0.25
pM_S = 0.4
pM_M = 0.75

#Offsets for first time index for each car
#Starts with car 1, goes up to 6
time_offsets = [0, 1, 5, 12, 4, 17]

#Time step for nuscene data
dt = 0.5

def load_data(filename):
    """Load in the data from the csv log

    Parameters:
    filename (str)  -- the name of the csv log

    Returns:
    data (float list)   -- the logged car data
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

    # Fixing a glitch in importing the time header
    for key in data:
        if "Time" in key:
            data["Time"] = data.pop(key)
            break

    # Add Speed data to the Dictionary
    for j in range(1,7):    #Loop through i=1 to i=6
        x = []
        y = []
        xname = "X_" + str(j)
        yname = "Y_" + str(j)
        sname = "S_" + str(j)
        data[sname] = []
        x = data[xname]
        y = data[yname]
        for i in range(len(x)-2):
            data[sname].append(math.sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2)/dt)

    return data

def hist_plotter(data, car_num, dt):
    """Takes in speed data and list of car numbers, and makes a histogram
        of all their speeds
    """
    speed = []

    for i in range(len(car_num)):
        sname = "S_" + str(car_num[i])
        speed_dat = data[sname]
        for j in range(len(data[sname])):
            speed.append(speed_dat[j])

    numbins = int(len(speed)/3)
    plt.hist(speed, bins=numbins)
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Frequency")


def sensor_model_stopped(data, car_num, dt):
    """ Uses car data to create a histogram of vehicle
        speed and then creates a pdf for a stopped car
    """
    speed = []

    for i in range(len(car_num)):
        sname = "S_" + str(car_num[i])
        speed_dat = data[sname]
        for j in range(len(data[sname])):
            speed.append(speed_dat[j])

    # Plot histogram
    numbins = int(len(speed)/2)
    plt.hist(speed, bins=numbins)

    # Fit speeds with a normal distribution
    mu, std = norm.fit(speed)

    # Make piecewise probability distribution function
    # Uniform distribution between 0 and mu calculated for normal
    # dist. After that, just a half norm
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, len(speed))
    p = []
    for i in range(len(x)):
        if (x[i] < mu):
            p.append(norm(mu, std).pdf(mu))
        else:
            p.append(norm(mu, std).pdf(x[i]))

    plt.plot(x, p, 'k', linewidth=2)
    plt.xlim(0,.18)
    plt.title("Speed Histogram for Stopped Car Overlaid with Custom Dist.")
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Frequency")

    return [mu,std]

def sensor_model_moving(data, car_num, dt):
        """ Uses car data to create a histogram of vehicle
            speed and then create a pdf
            Calculate speed using distance from euclidean change in position
            returns the average and standard devation for gaussian fit of data
        """
        speed = []

        for i in range(len(car_num)):
            sname = "S_" + str(car_num[i])
            speed_dat = data[sname]
            for j in range(len(data[sname])):
                speed.append(speed_dat[j])

        # Plot histogram
        numbins = int(len(speed)/0.3)
        plt.hist(speed, bins=numbins, range=(0,12))

        # Fit speeds with a normal distribution
        mu, std = norm.fit(speed)

        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 5*len(speed))
        p = norm.pdf(x, mu, std)

        plt.plot(x, p, 'k', linewidth=2)
        plt.title("Speed Histogram for Moving Car Overlaid with Normal Dist.")
        plt.xlabel("Speed (m/s)")
        plt.ylabel("Frequency")
        return [mu,std]

def p_moving_s(s):
    """ Takes in car speed, returns p(s|moving), which is the probability
        that speed measurement is s if the car is moving
    """
    pdf_val = norm(MOVE_MU, MOVE_STD).pdf(s)
    # cdf integrates over pdf. Put in a high value of 10 to get whole range,
    # then subtract the region less than 0 because those speeds are impossible
    cdf_val = halfnorm(MOVE_MU, MOVE_STD).cdf(10)-norm(MOVE_MU, MOVE_STD).cdf(0) # normalize with cdf
    prob = pdf_val/cdf_val
    return prob

def p_stopped_s(s):
    """ Takes in car speed, returns p(s|stopped), which is the probability
        that speed measurement is s if the car is stopped
    """
    # Make piecewise probability distribution function
    # Uniform distribution between 0 and mu calculated for normal
    # dist. After that, just a half norm
    if (s < STOP_MU):
        pdf_val = norm(STOP_MU, STOP_STD).pdf(STOP_MU)
    else:
        pdf_val = norm(STOP_MU, STOP_STD).pdf(s)

    # cdf integrates over pdf. Put in a high value of 10 to get whole range,
    # then subtract the region less than mu and add in the uniform region
    cdf_val = norm(STOP_MU, STOP_STD).cdf(10) + STOP_MU*norm(STOP_MU, STOP_STD).pdf(STOP_MU) - norm(STOP_MU, STOP_STD).cdf(STOP_MU)
    prob = pdf_val/cdf_val # normalize with cdf
    return prob

def bayes_filter_step(b_x_tp_S, b_x_tp_M, s):
    """ Returns the belief (probability) bel_(x_t) for the moving and stopped
        states
        inputs: the previous belief in stopped and moving state, current speed
        output the predicted beliefs
    """
    #Prediction Step
    #bel_bar(x=S) = p(S|S)*p(S) + p(S|M)*p(M)
    bb_x_t_S = pS_S*b_x_tp_S + pS_M*b_x_tp_M
    bb_x_t_M = pM_S*b_x_tp_S + pM_M*b_x_tp_M

    #Correction step
    b_x_t_S = p_stopped_s(s)*bb_x_t_S
    b_x_t_M = p_moving_s(s)*bb_x_t_M

    #Normalize
    norm = b_x_t_S + b_x_t_M
    b_x_t_S = b_x_t_S/norm
    b_x_t_M = b_x_t_M/norm

    return [b_x_t_S, b_x_t_M]

def plot_bayes(data, time_offset, times):
    """ Plots the Bayes filter prediction for a given car's data
        vs. time
    """

    #Initialize beliefs for each state
    bf = []
    b_x_tp_S = 0.5
    b_x_tp_M = 0.5
    for i in range(len(data)):
        # Repeatedly calls Bayes filter step, then plots vs. time
        [b_x_tp_S, b_x_tp_M] = bayes_filter_step(b_x_tp_S, b_x_tp_M, data[i])
        bf.append(b_x_tp_S)

    plt.plot(times[time_offset:time_offset+len(data)], bf)

def main():
    """Run a 1D Bayes filter on logged movement """

    filename = "E205_Lab2_NuScenesData.csv"
    data = load_data(filename)

    # global variables
    global STOP_MU
    global STOP_STD
    global MOVE_MU
    global MOVE_STD

    # Use car 4 data to develop conditional stopped probabilities
    # p(s_i|x_i = stopped)
    plt.figure(1)
    [STOP_MU,STOP_STD] = sensor_model_stopped(data, [4], dt)
    plt.show()

    # Use car 1 to develop our model for a moving car
    # p(s_i|x_i = moving)
    plt.figure(2)
    [MOVE_MU,MOVE_STD] = sensor_model_moving(data, [1], dt)
    plt.show()

    # Make histogram for cars 2, 3, 5
    plt.figure(4)
    hist_plotter(data, [2,3,5], dt)
    plt.title("Speed Histogram for Cars 2, 3, and 5")
    plt.show()

    # Plot stopped probability for each car vs. time
    plt.figure(5)
    times = data["Time"]
    
    for i in range(1,4):
        sname = "S_" + str(i)
        speeds = data[sname]
        plt.subplot(3, 1, i)
        plt.ylim(-.1,1.1)
        plt.xlim(-1,20)
        plt.title("Car " + str(i))
        plt.xlabel("Time (s)")
        plt.ylabel("p(stopped)")
        plot_bayes(speeds, time_offsets[i-1], times)
    plt.show()

    plt.figure(6)
    for i in range(4,7):
        sname = "S_" + str(i)
        speeds = data[sname]
        plt.subplot(3, 1, i-3)
        plt.ylim(-.1,1.1)
        plt.xlim(-1,20)
        plt.title("Car " + str(i))
        plt.xlabel("Time (s)")
        plt.ylabel("p(stopped)")
        plot_bayes(speeds, time_offsets[i-1], times)
    plt.show()

    print("Exiting...")

    return 0


if __name__ == "__main__":
    main()
