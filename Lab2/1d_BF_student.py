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
from scipy.stats import norm, halfnorm
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

    #Add Speed data to the Dictionary
    for j in range(1,6):    #Loop through i=1 to i=5
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

def sensor_model_hist(data, car_num, dt):
    """Create a probability density function based on histogram
    """
    speed = []

    for i in range(len(car_num)):
        sname = "S_" + str(car_num[i])
        speed_dat = data[sname]
        for j in range(len(data[sname])):
            if (speed_dat[j] < 1.0):
                speed.append(speed_dat[j])
    numbins = int(len(speed)/3)
    hist = np.histogram(speed, bins=numbins, density=True)
    hist_dist = scipy.stats.rv_histogram(hist)
    #Can then call .pdf and .df on this type of object

    plt.hist(speed, bins=numbins, density=1)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, len(speed))
    p = hist_dist.pdf(x)
    
    plt.plot(x, p, 'k', linewidth=2)
    plt.title("Normalized Speed Histogram overlaid with PDF")
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Frequency (%)")

    
    return hist_dist

def sensor_model(data, car_num, dt):
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
            if (speed_dat[j] < 1.0):
                speed.append(speed_dat[j])

    mu, std = norm.fit(speed)
    numbins = int(len(speed)/3)
    plt.hist(speed, bins=numbins, density=1)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, len(speed))
    p = norm.pdf(x, mu, std)
    
    plt.plot(x, p, 'k', linewidth=2)
    plt.title("Normalized Speed Histogram overlaid with Normal Distriubition")
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Frequency (%)")

    return [mu,std]

def p_s_x_moving(s, mu, std):
    pdf_val = norm(mu, std).pdf(s)
    cdf_val = norm(mu, std).cdf(10)
    prob = pdf_val/cdf_val    
    return prob

def p_s_x_stopped(s, mu, std):
    pdf_val = norm(mu, std).pdf(s)
    cdf_val = norm(mu, std).cdf(10) 
    prob = pdf_val/cdf_val    
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
    b_x_t_S = p_s_x_stopped(s,STOP_MU, STOP_STD)*bb_x_t_S
    b_x_t_M = p_s_x_moving(s,MOVE_MU, MOVE_STD)*bb_x_t_M
    #Normalize
    norm = b_x_t_S + b_x_t_M
    b_x_t_S = b_x_t_S/norm
    b_x_t_M = b_x_t_M/norm

    return [b_x_t_S, b_x_t_M]

def plot_bayes(data, time_offset, times):
    #Initialize beliefs for each state
    bf = []
    b_x_tp_S = 0.5
    b_x_tp_M = 0.5
    for i in range(len(data)):
        [b_x_tp_S, b_x_tp_M] = bayes_filter_step(b_x_tp_S, b_x_tp_M, data[i])
        bf.append(b_x_tp_S)

    plt.plot(times[time_offset:time_offset+len(data)], bf)

def main():
    """Run a 1D Bayes filter on logged movement """

    filename = "E205_Lab2_NuScenesData.csv"
    data = load_data(filename)

    global STOP_MU
    global STOP_STD
    global MOVE_MU
    global MOVE_STD
    
    for key in data:
        print(key)

    #Use car 4 data to develop conditional stopped probabilities
    # p(s_i|x_i = stopped)
    plt.figure(1)
    [STOP_MU,STOP_STD] = sensor_model(data, [4], dt)
    plt.show()

    #Fit histogram distribution
    plt.figure(2)
    stop_hist_dist = sensor_model_hist(data, [4], dt)
    plt.show()

    #Use car 2,3,5 data for moving probability
    # p(s_i|x_i = moving)
    plt.figure(2)
    [MOVE_MU,MOVE_STD] = sensor_model(data, [2,3,5], dt)
    plt.show()

    #Fit histogram distribution
    plt.figure(4)
    move_hist_dist = sensor_model_hist(data, [2,3,5], dt)
    plt.show()

    times = data["Time (s)"]
    #bayes filter for each car
    for i in range(1,7):
        print(i)
        sname = "S_" + str(i)
        speeds = data[sname]
        if (i == 4):
            print(speeds)
        plt.figure(4+i)
        plot_bayes(speeds, time_offsets[i-1], times)
        plt.title(sname)
        plt.show()


    #In the overall scheme of things, need to include



    print("Exiting...")

    return 0


if __name__ == "__main__":
    main()
