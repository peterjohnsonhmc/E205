"""
Author: Peter Johnson and Pinky King
Email: pjohnson@g.hmc.edu, pking@g.hmc.edu
Date of Creation: 3/30/20
Description:
    Particle Filter implementation for localization of a robot in an underground mine
    This code is for teaching purposes for HMC ENGR205 System Simulation Final Project
"""

import csv
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import os.path
import scipy as sp
from scipy.stats import norm, uniform, multivariate_normal
from numpy import linalg as la
from statistics import stdev
from PIL import Image


NUM_PARTICLES = 50
global ax
L = 0.545 #m distance between center of Left and right wheels
N = 4 #3 state var + weight
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
#R_t = np.array([[0.014**2,0,0], [0,0.027**2,0], [0,0,0.002**2]])
R_t = np.array([[0.00435**2,0,0], [0,0.00435**2,0], [0,0,(0.00435/L)**2]])
VAR_UR = 0.005**2
VAR_UL = 0.005**2
VAR_THETA = (0.002/L)**2
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


def propagate_state(p_i_t, u_t, rand):
    """Propagate/predict the state based on chosen motion model
        Use the nonlinear function g(x_t_prev, u_t)

    Parameters:
    p_i_t (np.array)  -- the previous state estimate
    u_t (np.array)       -- the current control input (really is odometry)

    Returns:
    p_pred_t (np.array)   -- the predicted state
    """

    x,y,theta,w = p_i_t
    u_l, u_r = u_t

    #u_l += np.random.normal(0, np.sqrt(VAR_UL))
    #u_r += np.random.normal(0, np.sqrt(VAR_UR))

    if (rand):
        delta_d = (u_l + u_r)/2 + np.random.normal(0, 0.005) # stdev of 5 cm for forward movement
        delta_theta = (u_r - u_l)/L + np.random.normal(0, 0.15) # stdev of 5 deg for theta
    else:
        delta_d = (u_l + u_r)/2
        delta_theta = (u_r - u_l)/L

    p_pred_t = np.array([[x + delta_d*math.cos(theta+delta_theta/2)],
                         [y + delta_d*math.sin(theta+delta_theta/2)],
                         [wrap_to_pi(theta + delta_theta)],
                         [w]], dtype = np.float64)

    return p_pred_t.reshape((N,))

def subtractive_clustering(P_t):
    ra = 0.01 #3*np.sqrt(VAR_LIDAR) # neighborhood radius based on 99% confidence, which is 3 stdev
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
    return ck
    potck = pot[maxindex]
    potlast = 0
    centroids.append(ck)

    return centroids[0:-1]



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
    x, y, theta, w = x_t_pred
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
            flipped_azimuth = -(azimuth-math.pi/2)
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
    x, y, theta, w = sigma_point
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

def find_weight(p_i_t, z_t, azimuth_t, Q_t):

    z_bar = calc_expected_meas(p_i_t, azimuth_t)
    x, y, theta, w = p_i_t
    row, col = xy_to_grid(x,y)

    Q_t = Q_t
    #w_t = multivariate_normal.pdf(z_t.reshape((NUM_BEAMS,)),z_bar.reshape((NUM_BEAMS,)), Q_t)
    w_t = 0
    # for i in range(NUM_BEAMS):
    #
    #     curr_weight = norm.pdf(z_t[i], z_bar[i], Q_t[i,i])
    #     importance_dist = (norm.pdf(z_t[i], z_bar[i], 10*Q_t[i,i]))
    #     if importance_dist == 0:
    #         curr_weight = 0
    #     else:
    #         curr_weight = curr_weight/importance_dist
    #
    #     #print(curr_weight)
    #     #
    #     # if curr_weight < 1e-10:
    #     #     curr_weight = 1e-10
    #
    #     w_t += np.longdouble(curr_weight)
    for i in range(NUM_BEAMS):
        res = abs(z_bar[i] - z_t[i])
        perc_off = res/z_t[i]
        w_t += perc_off
    w_t = w_t/NUM_BEAMS
    w_t = 1/w_t

    if (in_bounds(row, col)==False or sum(MAP[col,row]) > 300):
        # this is a particle out of bounds, want 0 weight
        return 0.0

    return w_t

def prediction_step(P_prev, u_t, z_t, azimuth_t, Q_t):
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
        p_pred = propagate_state(p_prev, u_t, 1)
        # find particle's weight using wt = P(zt | xt)
        w_t = find_weight(p_pred, z_t, azimuth_t, Q_t)
        w_tot += w_t
        # add new particle to the current belief
        p_pred[N-1] = w_t
        p_pred.reshape((N,))
        P_pred.append(p_pred)

    # while (w_tot <= RTICLES*10e-80):
    #     w_tot = 0
    #     var_lidar *= 100
    #     var_theta *= 100
    #     for i in range(0, NUM_PARTICLES):
    #         p_pred = P_pred[i]
    #         z_g_t = local_to_global(p_pred, z_t)
    #         w_t = find_weight(p_pred, z_g_t, u_t, var_lidar, var_theta)
    #         w_tot += w_t
    #         p_pred[5] = w_t
    #         P_pred[i] = p_pred

    return [P_pred, w_tot]



def correction_step(P_pred, w_tot):
    """Compute the correction of EKF

    Parameters:
    P_pred    (list of np.array)  -- the predicted particles of time t
    w_tot     (np.double)             -- the sum of all the particle weights

    Returns:
    P_corr    (list of np.array)  -- the corrected particles of time t
    """
    P_corr = []

    p0 = P_pred[0]
    w0 = p0[N-1].copy()
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
            w_j = p_j[N-1].copy()
            wsum += w_j

        p_c = P_pred[j]
        p_c.reshape((N,))
        #print(p_c)
        P_corr.append(p_c)

    return P_corr

def simple_clustering(P_t):
    highest_weight = 0;
    best_particle = P_t[0];
    for p in P_t:
        if p[N-1] > highest_weight:
            highest_weight = p[N-1]
            best_particle = p
    return best_particle


def main():
    """Run a EKF on logged data from IMU and LiDAR moving in a box formation around a landmark"""

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


    NUM_BEAMS = int(263/3)
    correcting = []

    x_min, y_max = grid_to_xy(0,0)
    x_max, y_min = grid_to_xy(ROWS-1,COLS-1)

    P_prev_t = []
    for i in range(0,NUM_PARTICLES):
        randx = np.random.uniform(x_min, x_max)
        randy = np.random.uniform(y_min, y_max)
        randtheta = np.random.uniform(-math.pi,math.pi)
        #p = np.array([0, randx, 0, randy, randtheta, 1/NUM_PARTICLES], dtype = np.double)
        #p = np.array([0, 5, 0, -5, randtheta, 1/NUM_PARTICLES], dtype = np.double)
        #p = np.array([0, randx, 0, randy, 0, 0, 0, 1/NUM_PARTICLES], dtype = np.double) # initialize particles to all have the same weight
        p = np.array([0, 0, 0, 1/NUM_PARTICLES], dtype = np.double) # known start
        p.reshape((N,))
        P_prev_t.append(p)


    # Run filter over data
    

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
    # movements = ["01M"]
    time_step_movement =[]
    x_estimates = []
    y_estimates = []
    theta_estimates = []
    x_estimates_simp = []
    y_estimates_simp = []
    theta_estimates_simp = []

    state_estimates_encoder = []

    for movement in movements:
        
        print(movement)
        
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

        lidar_index = 1

        for t, _ in enumerate(encoder_times):

            #print(encoder_times[t])

            # Get control input
            u_t = np.array([[u_l[t]], [u_r[t]]], dtype = np.float64)
            # print("u_t: ", u_t.shape)

            # figure out if we will correct
            # while (z_range_var[lidar_index] >= 1):
            #     lidar_index += 1

            if (lidar_index < len(lidar_times)-NUM_BEAMS and lidar_times[lidar_index] > encoder_times[t] and lidar_times[lidar_index] < encoder_times[t+1]+0.9):

                ax.imshow(im2, extent=[low_x, high_x, high_y, low_y ])
                plt.xlim([x_t_est[0]-100,x_t_est[0]+100])
                plt.ylim([high_y, low_y])

                #print("Predicting and correcting")
                # Get measurement
                z_t = np.zeros((NUM_BEAMS, 1), dtype = np.float64)
                azimuth_t = np.zeros((NUM_BEAMS, 1), dtype = np.float64)
                Q_t = np.zeros((NUM_BEAMS, NUM_BEAMS), dtype = np.float64)

                for i in range(NUM_BEAMS):
                    z_t[i] = z_range_mu[lidar_index+i]
                    azimuth_t[i] = z_azimuth_mu[lidar_index+i]
                    Q_t[i,i] = z_range_var[lidar_index+i]
                    #draw_points(z_t[i], azimuth_t[i], x_t_pred[0], x_t_pred[1], x_t_pred[2])

                P_pred_t, w_tot = prediction_step(P_prev_t, u_t, z_t, azimuth_t, Q_t)

                P_est_t = correction_step(P_pred_t, w_tot)
                x_t_est = subtractive_clustering(P_est_t)
                x_t_est_simple = simple_clustering(P_est_t)


                lidar_index += NUM_BEAMS

                # x_t_est = propagate_state(x_t_prev, u_t)
                # sigma_t_est = R_t
                # x_t_est[2] = 0
                #plt.subplot(2,1,1)
                #plt.scatter(x_t_est_simple[0], x_t_est_simple[1], s=5, marker='.', color='b')
                #plt.xlim([low_x,int(high_x/5)])
                #plt.xlim([x_t_est[0]-100,x_t_est[0]+100])

                # for p in P_est_t:
                #     x = p[0]
                #     y = p[1]
                #     plt.scatter(x, y, c='r', marker='.')

                # plt.subplot(2,1,2)
                # plt.scatter(encoder_times[t], x_t_est[2], color='b')
                # plt.scatter(encoder_times[t], x_t_pred[2], color='r')
                # plt.xlim([55, 75])
                # plt.ylim([-math.pi, math.pi])

                #plt.pause(0.0001)
                #ax.clear()
                correcting.append(1)

            else:
                # we can't predict without correcting
                #print("Propagating only")

                P_est_t = list(map(lambda p_i_t: propagate_state(p_i_t, u_t, 0), P_prev_t))
                x_t_est = subtractive_clustering(P_est_t)
                x_t_est_simple = simple_clustering(P_est_t)

                correcting.append(0)

            time_step_movement.append(movement)
            x_estimates.append(x_t_est[0])
            y_estimates.append(x_t_est[1])
            theta_estimates.append(x_t_est[2])
            x_estimates_simp.append(x_t_est_simple[0])
            y_estimates_simp.append(x_t_est_simple[1])
            theta_estimates_simp.append(x_t_est_simple[2])

            P_prev_t = P_est_t


        print("Estimated state: \n", x_t_est)



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

    with open('500particles_fullrun.csv', 'w', newline='') as csvfile:
        fieldnames = ['times','movement', 'x_sub', 'y_sub', 'theta_sub', 'x_simp', 'y_simp', 'theta_simp', 'correcting']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for t, _ in enumerate(encoder_times):
            writer.writerow({'times': encoder_times[t],'movement': time_step_movement[t], 'x_sub': x_estimates[t], \
             'y_sub': y_estimates[t], 'theta_sub': theta_estimates[t], 'x_simp': x_estimates_simp[t], \
              'y_simp': y_estimates_simp[t], 'theta_simp': theta_estimates_simp[t], "correcting": correcting[t]})

    print("Done plotting, exiting")

    im.show()
    im.save("out.png")
    im.close()



if __name__ == "__main__":
    main()
