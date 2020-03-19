"""
Author: Peter Johnson and Pinky King
Email: pjohnson@g.hmc.edu, pking@g.hmc.edu
Based on code by Andrew Q. Pham apham@g.hmc.edu
Date of Creation: 2/26/20
Description:
    Extended Kalman Filter implementation to filtering localization estimate
    This code is for teaching purposes for HMC ENGR205 System Simulation Lab 3
"""

import csv
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import os.path

HEIGHT_THRESHOLD = 0.0          # meters
GROUND_HEIGHT_THRESHOLD = -.4      # meters
dt = 0.1                        # timestep seconds                    
X_L = 5.                          # Landmark position in global frame
Y_L = -5.                          # meters
EARTH_RADIUS = 6.3781E6          # meters


def load_data(filename):
    """Load data from the csv log

    Parameters:
    filename (str)  -- the name of the csv log

    Returns:
    data (dict)     -- the logged data with data categories as keys
                       and values list of floats
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
                data[h].append(float(element))
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
    lat_gps     (float)    -- latitude coordinate
    lon_gps     (float)    -- longitude coordinate
    lat_origin  (float)    -- latitude coordinate of your chosen origin
    lon_origin  (float)    -- longitude coordinate of your chosen origin

    Returns:
    x_gps (float)          -- the converted x coordinate
    y_gps (float)          -- the converted y coordinate
    """
    x_gps = EARTH_RADIUS*(math.pi/180.)*(lon_gps - lon_origin)*math.cos((math.pi/180.)*lat_origin)
    y_gps = EARTH_RADIUS*(math.pi/180.)*(lat_gps - lat_origin)

    return x_gps, y_gps


def wrap_to_pi(angle):
    """Wrap angle data in radians to [-pi, pi]

    Parameters:
    angle (float)   -- unwrapped angle

    Returns:
    angle (float)   -- wrapped angle
    """
    while angle >= math.pi:
        angle -= 2*math.pi

    while angle <= -math.pi:
        angle += 2*math.pi
    return angle


def propogate_state(x_t_prev, u_t):
    """Propogate/predict the state based on chosen motion model
        Use the nonlinear function g(x_t_prev, u_t)

    Parameters:
    x_t_prev (np.array)  -- the previous state estimate
    u_t (np.array)       -- the current control input (really is odometry)

    Returns:
    x_bar_t (np.array)   -- the predicted state
    """
    #Destructure array
    xd, x, yd, y, thetad, theta, thetap = x_t_prev
    ux, uy, yaw = u_t


    x_bar_t = np.array([[xd + (ux*math.cos(yaw) - uy*math.sin(yaw))* dt],
                        [x  +  xd*dt],
                        [yd + (ux*math.sin(yaw) + uy*math.cos(yaw))* dt],
                        [y  +  yd*dt],
                        [(theta - thetap)/dt],
                        [yaw],
                        [theta]], dtype = float)


    #print("x_bar_t: ", x_bar_t.shape)

    return x_bar_t


def calc_prop_jacobian_x(x_t_prev, u_t):
    """Calculate the Jacobian of your motion model with respect to state

    Parameters:
    x_t_prev (np.array) -- the previous state estimate
    u_t (np.array)      -- the current control input (really is odometry)

    Returns:
    G_x_t (np.array)    -- Jacobian of motion model wrt to x
    """
    xd, x, yd, y, thetad, theta, thetap = x_t_prev
    ux, uy, yaw = u_t

    G_x_t = np.array([[  1, 0,  0, 0,  0,  0,        0],
                      [ dt, 1,  0, 0,  0,  0,        0],
                      [  0, 0,  1, 0,  0,  0,        0],
                      [  0, 0, dt, 1,  0,  0,        0],
                      [  0, 0,  0, 0,  0,  1/dt, -1/dt],
                      [  0, 0,  0, 0,  0,  0,        0],
                      [  0, 0,  0, 0,  0,  1,        0]],
                      dtype = float)  # add shape of matrix

    #print("G_x_t: ", G_x_t.shape)

    return G_x_t


def calc_prop_jacobian_u(x_t_prev, u_t):
    """Calculate the Jacobian of motion model (g) with respect to control input (u)

    Parameters:
    x_t_prev (np.array)     -- the previous state estimate
        in order: x' x  y' y theta' theta theta_prev
    u_t (np.array)          -- the current control input (really is odometry)

    Returns:
    G_u_t (np.array)        -- Jacobian of motion model wrt to u
    """

    xd, x, yd, y, thetad, theta, thetap = x_t_prev
    ux, uy, yaw = u_t


    G_u_t = np.array([[ dt*math.cos(yaw), -dt*math.sin(yaw),  (-ux*math.sin(yaw) - uy*math.cos(yaw))*dt],
                      [ 0,                 0,                                                         0],
                      [ dt*math.sin(yaw),  dt*math.cos(yaw),   (ux*math.cos(yaw) - uy*math.sin(yaw))*dt],
                      [ 0,                 0,                                                         0],
                      [ 0,                 0,                                                         0],
                      [ 0,                 0,                                                         1],
                      [ 0,                 0,                                                         0]],
                      dtype = float)  # add shape of matrix

    #print("G_u_t: ", G_u_t.shape)

    return G_u_t


def prediction_step(x_t_prev, u_t, sigma_x_t_prev):
    """Compute the prediction of EKF

    Parameters:
    x_t_prev (np.array)         -- the previous state estimate
    u_t (np.array)              -- the control input
    sigma_x_t_prev (np.array)   -- the previous variance estimate

    Returns:
    x_bar_t (np.array)          -- the predicted state estimate of time t
    sigma_x_bar_t (np.array)    -- the predicted variance estimate of time t
    """

    # Covariance matrix of control input
    #variance for ddx ddy, yaw
    R_t = np.array([[1.8373, 0,      0],
                    [0,      1.1991, 0],
                    [0,      0,      0.00058709]],
                    dtype=float)

    # Jacobians
    G_x_t = calc_prop_jacobian_x(x_t_prev, u_t)
    G_u_t = calc_prop_jacobian_u(x_t_prev, u_t)

    x_bar_t = propogate_state(x_t_prev, u_t)
    sigma_x_bar_t = G_x_t.dot(sigma_x_t_prev).dot(np.transpose(G_x_t)) + G_u_t.dot(R_t).dot(np.transpose(G_u_t))

    #Ensure Covariance is 7x7 matrix - needed to use .dot() operator
    #print("x_bar_t: ", x_bar_t.shape)
    #print("sigma_x_bar_t: ",sigma_x_bar_t.shape)

    return [x_bar_t, sigma_x_bar_t]


def calc_meas_jacobian(x_bar_t):
    """Calculate the Jacobian of your measurment model (h) with respect to state

    Parameters:
    x_bar_t (np.array)  -- the predicted state

    Returns:
    H_t (np.array)      -- Jacobian of measurment model
    """
    xd, x, yd, y, thetad, theta, thetap = x_bar_t

    H_t = np.array([[ 0, -1, 0,  0, 0, 0, 0],
                    [ 0,  0, 0, -1, 0, 0, 0]],
                    dtype = float)

    #print("H_t: ", H_t.shape)
    return H_t


def calc_kalman_gain(sigma_x_bar_t, H_t):
    """Calculate the Kalman Gain

    Parameters:
    sigma_x_bar_t (np.array)  -- the predicted state covariance matrix
    H_t (np.array)            -- the measurement Jacobian

    Returns:
    K_t (np.array)            -- Kalman Gain
    """

    # Covariance matrix of measurments
    #variance of lidar x, lidar y
    Q_t = np.array([[0.0075**2, 0],
                    [0, 0.0075**2]], dtype = float)

    H_t_T = np.transpose(H_t)

    K_t = sigma_x_bar_t.dot(H_t_T).dot(np.linalg.inv(H_t.dot(sigma_x_bar_t).dot(H_t_T) + Q_t))
    #Verify Kalman gain is 7x3 to go from measurement 3x1 to state size 7x1
    #print("Kalman Gain: ", K_t.shape)

    return K_t

def local_to_global(x_bar_t, z_t):
    """Rotate the lidar x and y measurements from the lidar frame to the global frame orientation

       Parameters:
       x_bar_t (np.array)  -- the predicted state
       z_t     (np.array)  -- the measurement vector in the lidar frame

       Returns:
       z_global (np.array) -- global orientation measurment vector
    """    
    xd, x, yd, y, thetad, theta, thetap = x_bar_t
    zx, zy = z_t
    w_theta = wrap_to_pi(-theta+math.pi/2)

    z_global = np.array([zx*math.cos(w_theta) + zy*math.sin(w_theta),
                         -zx*math.sin(w_theta) + zy*math.cos(w_theta)],
                         dtype = float)

    return z_global


def calc_meas_prediction(x_bar_t):
    """Calculate predicted measurement based on the predicted state
        Implements the nonlinear measrument h function
        Parameters:
        x_bar_t (np.array)  -- the predicted state

        Returns:
        z_bar_t (np.array)  -- the predicted measurement
    """

    xd, x, yd, y, thetad, theta, thetap = x_bar_t

    z_bar_t = np.array([X_L-x,
                        Y_L-y],
                        dtype = float)

    #print("z_bar_t: ", z_bar_t.shape)
    #print("z_bar_t: ", z_bar_t)

    return z_bar_t


def correction_step(x_bar_t, z_t, sigma_x_bar_t):
    """Compute the correction of EKF

    Parameters:
    x_bar_t       (np.array)    -- the predicted state estimate of time t
    z_t           (np.array)    -- the measured state of time t
    sigma_x_bar_t (np.array)    -- the predicted variance of time t

    Returns:
    x_est_t       (np.array)    -- the filtered state estimate of time t
    sigma_x_est_t (np.array)    -- the filtered variance estimate of time t
    """


    H_t = calc_meas_jacobian(x_bar_t)
    K_t = calc_kalman_gain(sigma_x_bar_t, H_t)

    z_bar_t = calc_meas_prediction(x_bar_t)
    resid = np.array([z_t[0]-z_bar_t[0],
                      z_t[1]-z_bar_t[1]],
                      dtype = float)
    #print("resid = ", resid)


    x_est_t = x_bar_t + K_t.dot(resid)
    sigma_x_est_t = (np.eye(7, dtype=float)-K_t.dot(H_t)).dot(sigma_x_bar_t)

    #Need to reshape to 1D array for later processing
    x_est_t = x_est_t.reshape((7,))
    #x_est_t = x_bar_t.reshape((7,))
    #Verify sizes
    #print("x_est_t: ", x_est_t.shape)
    #print("sigma_x_est_t: ",sigma_x_est_t.shape)

    #x_est_t = x_bar_t.reshape((7,))
    return [x_est_t, sigma_x_est_t]

def distance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist 


def path_rmse(state_estimates):
    """ Computes the RMSE error of the distance at each time step from the expected path

        Parameters:
        x_estimate      (np.array)    -- array  of state estimates

        Returns:
        rmse              (float)          -- rmse
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
    N = 7 # number of states
    #Start in NW corner
    state_est_t_prev = np.array([0,0,0,0,0,0,0])
    var_est_t_prev = np.identity(N)
    skip_est_t_prev = np.array([0,0,0,0,0,0,0])
    vskip_est_t_prev = np.identity(N)

    #allocate
    state_predictions = np.empty((N, len(time_stamps)))
    state_estimates = np.empty((N, len(time_stamps)))
    skipp_estimates = np.empty((N, len(time_stamps)))
    covariance_estimates = np.empty((N, N, len(time_stamps)))
    covarskipp_estimates = np.empty((N, N, len(time_stamps)))
    gps_estimates = np.empty((2, len(time_stamps)))

    #Added functionality to include correction not every time step
    skip = False


    #  Run filter over data
    for t, _ in enumerate(time_stamps):

        # Get control input
        u_t = np.array([[x_ddot[t]],
                        [y_ddot[t]],
                        [yaw_lidar[t]]])
        #print("u_t: ", u_t.shape)

        # Prediction Step
        state_pred_t, var_pred_t = prediction_step(state_est_t_prev, u_t, var_est_t_prev)
        skip_pred_t, vski_pred_t = prediction_step(skip_est_t_prev, u_t, vskip_est_t_prev)

        # Get measurement
        z_t = np.array([[x_lidar[t]], [y_lidar[t]]])
        #print("z_t: ", z_t.shape)
        z_t = local_to_global(state_pred_t, z_t)

        # Correction Step
        state_est_t, var_est_t = correction_step(state_pred_t, z_t, var_pred_t)
        if skip:
            skip_est_t, vski_est_t = skip_pred_t, vski_pred_t
            skip_est_t = skip_est_t.reshape((7,))
            skip = False
        else:
            skip_est_t, vski_est_t = correction_step(skip_pred_t, z_t, vski_pred_t)
            skip = True


        #  For clarity sake/teaching purposes, we explicitly update t->(t-1)
        state_est_t_prev = state_est_t
        var_est_t_prev = var_est_t

        skip_est_t_prev = skip_est_t
        vskip_est_t_prev = vski_est_t

        # Log Data
        state_estimates[:, t] = state_est_t
        state_predictions[:, t] = state_pred_t.reshape((7,))
        covariance_estimates[:, :, t] = var_est_t
        skipp_estimates[:, t] = skip_est_t
        covarskipp_estimates[:,:, t] = vski_est_t

        x_gps, y_gps = convert_gps_to_xy(lat_gps=lat_gps[t],
                                         lon_gps=lon_gps[t],
                                         lat_origin=lat_origin,
                                         lon_origin=lon_origin)
        gps_estimates[:, t] = np.array([x_gps, y_gps])



    #Plot or print results here
    print("\n\nDone filtering...plotting...")


    #Expected path
    pathx = [0,10,10,0,0]
    pathy = [0,0,-10,-10,0] 
    time_enum =  np.linspace(0,len(time_stamps),len(time_stamps))


    #RMSE
    rmse, residuals, mean_error, rmse_time = path_rmse(state_estimates)
    print("Expected Path RMSE: ", rmse)
    print(len(residuals))
    print(len(time_enum))
    print(residuals[0])
    print(time_enum[0])
    #print(residuals.shape)
    #print(time_enum.shape)

    # plt.figure(1)
    # plt.plot(time_enum, residuals)
    # plt.plot([0,len(time_enum)],[rmse, rmse])
    # #plt.plot([0,len(time_enum)],[mean_error, mean_error])
    # plt.plot(time_enum, rmse_time)
    # plt.xlabel('Time Steps')
    # plt.ylabel('Path Tracking Error [m]')
    # plt.legend(('Error','Overall RMSE','RMSE over time'))
    # plt.show()

    # plt.figure(1)
    # plt.axis([-5, 15, -15, 5])
    # plt.plot(pathx,pathy)
    # for t in range(len(time_stamps)):
    #     x = state_estimates[1][t]
    #     y = state_estimates[3][t]
    #     xg = gps_estimates[0][t]
    #     yg = gps_estimates[1][t]
    #     plt.scatter(x, y, c='r')
    #     plt.scatter(xg, yg, c='b')
    #     plt.pause(0.0005)
    # plt.show()

    labels = {0: "\u03C3\u2093\u1d65\u00b2 ((m/s)\u00b2)", 1: "\u03C3\u2093\u00b2 (m\u00b2)", 2: "\u03C3\u1d67\u1d65\u00b2 ((m/s)\u00b2)", 3: "\u03C3\u1d67\u00b2 (m\u00b2)", 4: "\u03C3\u209C\u1d65\u00b2 ((rad/s)\u00b2)", 5: "\u03C3\u209C\u00b2 (rad\u00b2)", 6: "\u03C3\u209C\u209A\u00b2 (rad\u00b2)"}
    plt.figure(1)
    for i in [4,5,6]:
        plt.subplot(3, 1, i-3)
        plt.plot(time_enum[3:50], covariance_estimates[i][i][3:50], marker="^", c='b')
        plt.plot(time_enum[3:50], covarskipp_estimates[i][i][3:50], marker=".", c='r')
        plt.ylim([.8*min(covariance_estimates[i][i][3:50]), 1.1*max(covarskipp_estimates[i][i][3:50])])
        plt.ylabel(labels[i])
        print(i)
    plt.xlabel("Time Steps")
    plt.show()
    plt.legend("Correction", "Less Correction")

    #Plot raw data and estimate
    plt.figure(2)
    plt.suptitle("EKF Localization: X & Y Measurements")
    #Plot x,y
    T1 = 0
    T2 = 150#len(time_stamps)
    #print(state_predictions[0][T1:T2])
    #print(np.mean(state_predictions[0][T1:T2]))
    plt.scatter(state_estimates[1][T1:T2], state_estimates[3][T1:T2], marker = '.')
    plt.scatter(gps_estimates[0][:],gps_estimates[1][:], marker = '.')
    plt.scatter(state_predictions[1][T1:T2], state_predictions[3][T1:T2], marker = '.')
    plt.plot(pathx,pathy)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.xlim([-10, 20])
    plt.ylim([-20, 10])
    plt.show()

    plt.figure(3)
    #plt.suptitle("EKF Localization: Est &  Meas Yaw")
    #Plot yaw
    #plt.scatter(time_enum, data["Yaw"], marker = '.')
    plt.scatter(time_enum, state_estimates[5][:], marker = '.')
    #plt.legend(['Yaw', 'Yaw Estimated'])
    plt.ylim([-math.pi-.2, math.pi+.2])
    plt.xlabel('Time Steps')
    plt.ylabel('Yaw Angle [radians]')
    plt.show()

    plt.figure(4)
    plt.suptitle("EKF Localization: X ")
    #Plot x,y
    plt.scatter(time_stamps, state_estimates[1][:],marker = '.')
    plt.scatter(time_stamps, state_predictions[1][:],marker = '.')
    plt.ylabel('X [m]')
    plt.ylim([-20, 10])
    plt.show()

    plt.figure(5)
    plt.suptitle("EKF Localization: Y ")
    #Plot x,y
    plt.scatter(time_stamps, state_estimates[3][:])
    plt.scatter(time_stamps, state_predictions[3][:])
    plt.ylabel('Y [m]')
    plt.ylim([-20, 10])
    plt.show()

    plt.figure(6)
    plt.suptitle("EKF Localization: Z_bar_t x ")
    #Plot x,y
    plt.scatter(time_stamps, z_bars[0][:])
    plt.scatter(time_stamps, x_lidar[:])
    plt.ylabel('Xmeasbar [m]')
    plt.ylim([0, 10])
    plt.show()

    plt.figure(7)
    plt.suptitle("EKF Localization: Z_bar_t y ")
    #Plot x,y
    plt.scatter(time_stamps, z_bars[1][:])
    plt.scatter(time_stamps, y_lidar[:])
    plt.ylabel('Ymeasbar [m]')
    plt.ylim([-10, 10])
    plt.show()

    plt.figure(8)
    plt.suptitle("V_x")
    plt.scatter(time_stamps, state_estimates[0][:])
    plt.show()

    plt.figure(9)
    plt.suptitle("V_y")
    plt.scatter(time_stamps, state_estimates[2][:])
    plt.show()


    print("Exiting...")

    return 0


if __name__ == "__main__":
    main()
