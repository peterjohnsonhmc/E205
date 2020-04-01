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

HEIGHT_THRESHOLD = 0.0          # meters
GROUND_HEIGHT_THRESHOLD = -.4      # meters
dt = 0.1                        # timestep seconds
X_L = 5.                          # Landmark position in global frame
Y_L = -5.                          # meters
EARTH_RADIUS = 6.3781E6          # meters
NUM_PARTICLES = 300
# variances
VAR_AX = 1.8373
VAR_AY = 1.1991
VAR_THETA = 0.00058709
VAR_LIDAR = .01 #0.0075**2 # this is actually range but works for x and y

PRINTING = False


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

    #Convert from CW degrees to CCW radians
    for i in range(0, len(data["Yaw"])):
        theta = data["Yaw"][i]
        theta = -theta*2*math.pi/360
        theta = wrap_to_pi(theta)
        data["Yaw"][i] = theta

    return data, is_filtered


def save_data(data, filename):
    """Save data from dictionary to csv

    Parameters:
    filename (str)  -- the name of the csv log
    data (dict)     -- data to log
    """
    header = ["X", "Y", "Z", "Time Stamp", "Latitude", "Longitude",
              "Yaw", "Pitch", "Roll", "AccelX", "AccelY", "AccelZ"]
    f = open(filename, "w")
    num_rows = len(data["X"])
    for i in range(num_rows):
        for h in header:
            f.write(str(data[h][i]) + ",")

        f.write("\n")

    f.close()


def filter_data(data):
    """Filter lidar points based on height and duplicate time stamp

    Parameters:
    data (dict)             -- unfilterd data

    Returns:
    filtered_data (dict)    -- filtered data
    """

    # Remove data that is not above a height threshold to remove
    # ground measurements and remove data below a certain height
    # to remove outliers like random birds in the Linde Field (fuck you birds)
    filter_idx = [idx for idx, ele in enumerate(data["Z"])
                  if ele > GROUND_HEIGHT_THRESHOLD and ele < HEIGHT_THRESHOLD]

    filtered_data = {}
    for key in data.keys():
        filtered_data[key] = [data[key][i] for i in filter_idx]

    # Remove data that at the same time stamp
    ts = filtered_data["Time Stamp"]
    filter_idx = [idx for idx in range(1, len(ts)) if ts[idx] != ts[idx-1]]
    for key in data.keys():
        filtered_data[key] = [filtered_data[key][i] for i in filter_idx]

    return filtered_data


def convert_gps_to_xy(lat_gps, lon_gps, lat_origin, lon_origin):
    """Convert gps coordinates to cartesian with equirectangular projection

    Parameters:
    lat_gps     (np.double)    -- latitude coordinate
    lon_gps     (np.double)    -- longitude coordinate
    lat_origin  (np.double)    -- latitude coordinate of your chosen origin
    lon_origin  (np.double)    -- longitude coordinate of your chosen origin

    Returns:
    x_gps (np.double)          -- the converted x coordinate
    y_gps (np.double)          -- the converted y coordinate
    """
    x_gps = EARTH_RADIUS*(math.pi/180.)*(lon_gps - lon_origin)*math.cos((math.pi/180.)*lat_origin)
    y_gps = EARTH_RADIUS*(math.pi/180.)*(lat_gps - lat_origin)

    return x_gps, y_gps


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


def propogate_state(p_i_t, u_t):
    """Propogate/predict the state based on chosen motion model
        Use the nonlinear function g(x_t_prev, u_t)

    Parameters:
    x_t_prev (np.array)  -- the previous state estimate
    u_t (np.array)       -- the current control input (really is odometry)

    Returns:
    x_bar_t (np.array)   -- the predicted state
    """
    #Destructure array
    xd, x, yd, y, thetad, theta, thetap, w = p_i_t

    ux, uy, yaw = u_t
    ux += np.random.normal(0, np.sqrt(VAR_AX))
    uy += np.random.normal(0, np.sqrt(VAR_AY))
    yaw += np.random.normal(0, np.sqrt(VAR_THETA))

    p_pred_t = np.array([xd + (ux*math.cos(yaw) - uy*math.sin(yaw))* dt,
                        x  +  xd*dt,
                        yd + (ux*math.sin(yaw) + uy*math.cos(yaw))* dt,
                        y  +  yd*dt,
                        wrap_to_pi(theta - thetap)/dt,
                        yaw,
                        theta,
                        w], dtype = np.double)


    #print("x_bar_t: ", x_bar_t.shape)

    return p_pred_t

def calc_meas_prediction(p_i_t):
    """Calculate predicted measurement based on the predicted state
        Implements the nonlinear measrument h function
        Parameters:
        x_bar_t (np.array)  -- the predicted state

        Returns:
        z_bar_t (np.array)  -- the predicted measurement
    """

    xd, x, yd, y, thetad, theta, thetap, w = p_i_t

    z_bar_t = np.array([X_L-x,
                        Y_L-y],
                        dtype = np.double)

    #print("z_bar_t: ", z_bar_t.shape)
    #print("z_bar_t: ", z_bar_t)

    return z_bar_t

def longpdf(mu, var, x):
    const = sp.longdouble(1.0/np.sqrt(2*math.pi*var))
    exponent = sp.longdouble(-0.5*((x-mu)**2)/var)
    return sp.longdouble(const*np.exp(exponent))

def find_weight(p_i_t, z_t, u_t):
    z_x, z_y = z_t
    _, _, u_theta = u_t

    z_bar_x, z_bar_y = calc_meas_prediction(p_i_t)
    _, _, _, _, _, theta, _, _ = p_i_t
    if (PRINTING):
        print("z_x: ", z_x)
        print("z_y: ", z_y)
        print("z_bar_x: ", z_bar_x)
        print("z_bar_y: ", z_bar_y)

    pdf_val = longpdf(z_bar_x, VAR_LIDAR, z_x)
    w_xt = sp.longdouble(pdf_val)

    pdf_val = longpdf(z_bar_y, VAR_LIDAR, z_y)
    w_yt = sp.longdouble(pdf_val)

    pdf_val = longpdf(u_theta, VAR_THETA, theta)
    w_theta = sp.longdouble(pdf_val)

    if (PRINTING):
        print("w_yt: ", w_yt)
        print("w_xt: ", w_xt)

    if (w_yt == 0.0):
        w_yt = 10e-20
    if (w_xt == 0.0):
        w_xt = 10e-20
    if (w_theta == 0.0):
        w_theta = 10e-20

    return sp.longdouble(w_xt*w_yt*w_theta)

def local_to_global(p_i_t, z_t):
    """Rotate the lidar x and y measurements from the lidar frame to the global frame orientation

       Parameters:
       x_bar_t (np.array)  -- the predicted state
       z_t     (np.array)  -- the measurement vector in the lidar frame

       Returns:
       z_global (np.array) -- global orientation measurment vector
    """
    xd, x, yd, y, thetad, theta, thetap, w = p_i_t
    zx, zy = z_t
    w_theta = wrap_to_pi(-theta+math.pi/2)

    z_global = np.array([zx*math.cos(w_theta) + zy*math.sin(w_theta),
                         -zx*math.sin(w_theta) + zy*math.cos(w_theta)],
                         dtype = np.double)

    return z_global



def prediction_step(P_prev, u_t, z_t):
    """Compute the prediction of EKF

    Parameters:
    P_prev (list of np.arrays)  -- previous particles
    u_t (np.array)              -- the control input
    z_t (np.array)              -- the measurement

    Returns:
    P_pred                      -- the predicted particles
    """

    P_pred = []
    w_tot = 0
    # for loop over all of the previous particles
    for p_prev in P_prev:
        # find new state given previous particle, odometry + randomness (motion model)
        p_pred = propogate_state(p_prev, u_t)
        # Globalize the measurment for each particle
        z_g_t = local_to_global(p_pred, z_t)
        # find particle's weight using wt = P(zt | xt)
        w_t = find_weight(p_pred, z_g_t, u_t)
        w_tot += w_t
        # add new particle to the current belief
        p_pred[7] = w_t
        p_pred.reshape((8,))
        P_pred.append(p_pred)


    return [P_pred, w_tot]



def correction_step(P_pred, w_tot):
    """Compute the correction of EKF

    Parameters:
    P_pred    (list of np.array)  -- the predicted particles of time t
    w_tot     (np.double)             -- the sum of all the particle weights

    Returns:
    P_corr    (list of np.array)  -- the corrected particles of time t
    """
    if (PRINTING):
        print("RESAMPLING")
        for p in P_pred:
            print("x: ", p[1], "   y: ", p[3], "   weight: ", p[7])

    P_corr = []

    p0 = P_pred[0]
    w0 = p0[7].copy()
    # resampling algorithm
    for p in P_pred:
        r = np.random.uniform(0, 1)*w_tot
        j = 0
        wsum = w0.copy()
        while (wsum < r):
            j += 1
            if (j == NUM_PARTICLES-1):
                break
            p_j = P_pred[j]
            w_j = p_j[7].copy()
            wsum += w_j

        p_c = P_pred[j]
        p_c.reshape((8,))
        #print(p_c)
        P_corr.append(p_c)

    return P_corr


def distance(x1,y1,x2,y2):
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
     return dist

def subtractive_clustering(P_t):
    ra = 3*np.sqrt(VAR_LIDAR) # neighborhood radius based on 99% confidence, which is 3 stdev
    pot = []
    maxpot = 0
    maxindex = 0
    centroids = []
    for i in range(0, NUM_PARTICLES):
        newpot = 0
        for j in range(0, NUM_PARTICLES):
            newpot += np.exp(-la.norm(P_t[i]-P_t[j],2)/(0.5*ra)**2)
        pot.append(newpot)
        if (newpot > maxpot):
            maxpot = newpot
            maxindex = i

    # Define first centroid center c1
    k = 1
    ck = P_t[maxindex]
    potck = pot[maxindex]
    potlast = 0
    centroids.append(ck)
    while (potck > 0.95*potlast):
        #Update Potential Values
        newpot = 0
        maxpot = 0
        maxindex = 0
        for i in range(0, NUM_PARTICLES):
            pot[i] -= potck*np.exp(-la.norm(P_t[i] - ck,2)/(0.75*ra)**2)
            if (pot[i] > maxpot):
                maxpot = pot[i]
                maxindex = i
        #Calculate kth centroid
        ck = P_t[maxindex]
        potlast = potck
        potck = pot[maxindex]
        centroids.append(ck)
        k = k + 1

    return centroids[0:-1]

def path_rmse(state_estimates):
    """ Computes the RMSE error of the distance at each time step from the expected path

        Parameters:
        x_estimate      (np.array)    -- array  of state estimates

        Returns:
        rmse              (np.double)          -- rmse
        residuals         (np.array)      -- array of residuals
    """
    x_est = state_estimates[1][:]
    y_est = state_estimates[3][:]
    sqerrors = []
    errors = []
    residuals = []
    rmse_time = []

    #resid = measured - predicted by segments

    for i in range(len(x_est)):
        if (x_est[i]<0 and y_est[i]>0):
            #Upper left corner
            resid = distance(x_est[i], y_est[i], 0,0)
            sqerror = resid**2

        elif (x_est[i]>10 and y_est[i]>0):
            #Upper right coner
            resid = distance(x_est[i], y_est[i], 10,0)
            sqerror = resid**2

        elif (x_est[i]>10 and y_est[i]<-10):
            #Lower right coner
            resid = distance(x_est[i], y_est[i], 10,-10)
            sqerror = resid**2

        elif (x_est[i]<0 and y_est[i]<-10):
            #Lower right coner
            resid = distance(x_est[i], y_est[i], 0,-10)
            sqerror = resid**2

        else:
            #General case
            r1 = (y_est[i] - 0)
            r2 = (x_est[i] - 10)
            r3 = (y_est[i] - (-10))
            r4 = (x_est[i] -0)
            resid = min(abs(r1),abs(r2),abs(r3),abs(r4))

        residuals.append(resid) #residuals are basically cte
        sqerrors.append(resid**2)
        errors.append(abs(resid))
        mse = np.mean(sqerrors)
        rmse = math.sqrt(mse)
        rmse_time.append(rmse)

    mean_error = np.mean(errors)
    mse = np.mean(sqerrors)
    rmse = math.sqrt(mse)

    return rmse, residuals, mean_error, rmse_time



def main():
    """Run a EKF on logged data from IMU and LiDAR moving in a box formation around a landmark"""

    np.random.seed(10)

    filepath = ""
    filename =  "2020_2_26__16_59_7" #"2020_2_26__17_21_59"
    data, is_filtered = load_data(filepath + filename)

    # Save filtered data so don't have to process unfiltered data everytime
    if not is_filtered:
        f_data = filter_data(data)
        save_data(f_data, filepath+filename+"_filtered.csv")

    # Load data into variables
    x_lidar = data["X"]
    y_lidar = data["Y"]
    z_lidar = data["Z"]
    time_stamps = data["Time Stamp"]
    lat_gps = data["Latitude"]
    lon_gps = data["Longitude"]
    yaw_lidar = data["Yaw"]
    pitch_lidar = data["Pitch"]
    roll_lidar = data["Roll"]
    x_ddot = data["AccelX"]
    y_ddot = data["AccelY"]

    lat_origin = lat_gps[0]
    lon_origin = lon_gps[0]

    #  Initialize filter
    N = 8 # number of states
    #Start in NW corner
    P_prev_t = []
    for i in range(0,NUM_PARTICLES):
        randx = np.random.uniform(-5,15)
        randy = np.random.uniform(-15,5)
        randtheta = np.random.uniform(-math.pi,math.pi)
        #p = np.array([0, randx, 0, randy, 0, randtheta, 0, 1/NUM_PARTICLES], dtype = np.double) # initialize particles to all have the same weight
        p = np.array([0, 0, 0, 0, 0, 0, 0, 1/NUM_PARTICLES], dtype = np.double) # known start
        p.reshape((8,))
        P_prev_t.append(p)

    #allocate
    gps_estimates = np.empty((2, len(time_stamps)))

    #Expected path
    pathx = [0,10,10,0,0]
    pathy = [0,0,-10,-10,0]

    #Initialize animated plot
    plt.figure(1)
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    plt.axis([-5, 15, -15, 5])
    ax.set_ylabel("Y position (Global Frame, m)")
    ax.set_xlabel("X position (Global Frame, m)")
    ax.legend(["Expected Path", "Estimated Position", "GPS Position"], loc='center right')


    #  Run filter over data
    for t, _ in enumerate(time_stamps):
        x_gps, y_gps = convert_gps_to_xy(lat_gps=lat_gps[t],
                                 lon_gps=lon_gps[t],
                                 lat_origin=lat_origin,
                                 lon_origin=lon_origin)
        #plt.axis([-15, 25, -25, 15])

        plt.axis([-5, 15, -15, 5])
        plt.plot(pathx,pathy)
        for p in P_prev_t:
            x = p[1]
            y = p[3]
            ax.scatter(x, y, c='r', marker='.')
        plt.scatter(x_gps, y_gps, c='b', marker='.')
        centroids = subtractive_clustering(P_prev_t)
        for c in centroids:
            plt.scatter(c[1],c[3], c='k', marker='.')
        plt.pause(0.00001)
        ax.clear()

        if (PRINTING):
            print("Time Step: %d", t)
            for p in P_prev_t:
                print("x: ", p[1], "   y: ", p[3], "   weight: ", p[7])

        # Get control input
        u_t = np.array([x_ddot[t],
                        y_ddot[t],
                        yaw_lidar[t]])
        #print("u_t: ", u_t.shape)

        # Get measurement
        z_t = np.array([x_lidar[t], y_lidar[t]])
        #print("z_t: ", z_t.shape)

        # Prediction Step
        P_pred_t,  w_tot = prediction_step(P_prev_t, u_t, z_t)

        # Correction Step
        P_t = correction_step(P_pred_t, w_tot)



        #  For clarity sake/teaching purposes, we explicitly update t->(t-1)
        P_prev_t = P_t


    plt.show()


    print("Done plotting, exiting")
    return 0


if __name__ == "__main__":
    main()
