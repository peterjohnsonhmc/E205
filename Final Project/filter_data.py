import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statistics import stdev
import csv
import math


def quat_to_euler(qx, qy, qz, qw):
    # based on code from Wikipedia
    # yaw calculation
    siny = 2*(qw*qz + qx*qy)
    cosy = 1 - 2*(qy*qy + qz*qz)
    yaw = math.atan2(siny, cosy)
    return yaw

def filter_ground_truth():
    open_path = "data/"
    save_path = "filtered_data/"
    f = open(open_path + "scanPoseEstimates.dat", 'r')

    with open(save_path + "scanPoseEstimates_filtered.csv", 'w', newline='') as csvfile:
        fieldnames = ['pose_num', 'x_truth', 'y_truth', 'theta_truth']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for line in f:
            pose_num, qx, qy, qz, qw, x, y, z = line.split()
            qx = float(qx)
            qy = float(qy)
            qz = float(qz)
            qw = float(qw)
            x = float(x)
            y = float(y)
            theta_truth = quat_to_euler(qx, qy, qz, qw)
            writer.writerow({'pose_num': pose_num, 'x_truth': x, 'y_truth': y, 'theta_truth': theta_truth})
    return 0

def filter_lidar_multiple(movement):
    path = "filtered_data/" + movement + "/" + movement + "_"

    # Load data
    f = open(path + "lidar_filtered.dat", 'r')
    f.readline()

    with open(path + "lidar_multi.csv", 'w', newline='') as csvfile:
        fieldnames = ['times', 'theta', 'range']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        line = f.readline()
        print(line)
        time, theta, range = line.split(',')
        time = float(time)
        theta = float(theta)
        range = float(range)
        current_time = time
        theta_list = []
        range_list = []
        theta_list.append(theta)
        range_list.append(range)

        for line in f:
            time, theta, range = line.split(',')
            time = float(time)
            theta = float(theta)
            range = float(range)
            if time < current_time+0.095:
                theta_list.append(theta)
                range_list.append(range)
            else:
                # calculate mu and standard deviation of 1/3 of each list
                length = len(theta_list)
                third = int(length / 3)
                theta_list1 = theta_list[0:third]
                theta_list2 = theta_list[third+1:2*third]
                theta_list3 = theta_list[2*third+1:]

                range_list1 = range_list[0:third]
                range_list2 = range_list[third+1:2*third]
                range_list3 = range_list[2*third+1:]

                theta = [theta_list1, theta_list2, theta_list3]
                range = [range_list1, range_list2, range_list3]

                theta_mu = []
                range_mu = []
                theta_var = []
                range_var = []

                for i in range(3):
                    theta_mu[i] = np.mean(theta[i])
                    range_mu[i] = np.mean(range[i])
                    if (len(theta[i]) ==1):
                        theta_var[i] = 0
                    else:
                        theta_var = (stdev(theta[i], theta_mu[i]))**2
                    if (len(range[i])==1):
                        range_var[i] = 0
                    else:
                        range_var = (stdev(range[i], range_mu[i]))**2
                writer.writerow({'times': current_time, 'theta_mu1': theta_mu[0], 'theta_var1': theta_var[0], 'range_mu1': range_mu[0], 'range_var1': range_var[0],
                                                        'theta_mu2': theta_mu[1], 'theta_var2': theta_var[1], 'range_mu2': range_mu[1], 'range_var2': range_var[1],
                                                        'theta_mu3': theta_mu[2], 'theta_var3': theta_var[2], 'range_mu3': range_mu[2], 'range_var3': range_var[2]})

                current_time = time


    return 0



def filter_lidar_encoder(movement):

    open_path = "data/" + movement + "/" + movement + "_"
    save_path = "filtered_data/" + movement + "/" + movement + "_"
    # f = open(open_path + "encoder.dat", 'r')
    # f.readline()
    #
    # with open(save_path + "encoder_filtered.csv", 'w', newline='') as csvfile:
    #     fieldnames = ['times', 'left', 'right']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #
    #     first_read = f.readline()
    #     time, left, right= first_read.split()
    #     time = float(time)
    #     left = float(left)
    #     right = float(right)
    #     startingtime = time
    #
    #     time = time - startingtime
    #     writer.writerow({'times': time, 'left': left, 'right': right})
    #
    #     left_prev = left
    #     right_prev = right
    #
    #     for line in f:
    #         time, left, right= line.split()
    #         time = float(time)
    #         left = float(left)
    #         right = float(right)
    #
    #         time = time - startingtime
    #         left_d = left - left_prev
    #         right_d = right - right_prev
    #         writer.writerow({'times': time, 'left': left_d, 'right': right_d})
    #
    #         left_prev = left
    #         right_prev = right

    # Load data
    f = open(open_path + "lidar.dat", 'r')
    f.readline()

    with open(save_path + "lidar_filtered.csv", 'w', newline='') as csvfile:
        fieldnames = ['times', 'theta', 'range']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for line in f:
            time, x, y, z, i =line.split()
            time = float(time)
            x = float(x)
            y = float(y)
            z = float(z)

            if (z <= 0.15 and z > -0.15):
                time = time #- startingtime
                theta = math.atan2(y,x)-math.pi
                range = math.sqrt(x**2 + y**2)
                writer.writerow({'times': time, 'theta': theta, 'range': range})

    # Load data
    f = open(save_path + "lidar_filtered.csv", 'r')
    # get rid of header
    f.readline()

    with open(save_path + "lidar_gauss.csv", 'w', newline='') as csvfile:
        fieldnames = ['times', 'theta_mu', 'theta_var', 'range_mu', 'range_var']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        line = f.readline()
        print(line)
        time, theta, range = line.split(',')
        time = float(time)
        theta = float(theta)
        range = float(range)
        current_time = time
        theta_list = []
        range_list = []
        theta_list.append(theta)
        range_list.append(range)

        for line in f:
            time, theta, range = line.split(',')
            time = float(time)
            theta = float(theta)
            range = float(range)
            if time < current_time+0.095:
                theta_list.append(theta)
                range_list.append(range)
            else:
                # calculate mu and standard deviation
                theta_mu = np.mean(theta_list)
                range_mu = np.mean(range_list)
                if (len(theta_list) ==1):
                    theta_var = 0
                else:
                    theta_var = (stdev(theta_list, theta_mu))**2
                if (len(range_list)==1):
                    range_var = 0
                else:
                    range_var = (stdev(range_list, range_mu))**2
                writer.writerow({'times': current_time, 'theta_mu': theta_mu, 'theta_var': theta_var, 'range_mu': range_mu, 'range_var': range_var})
                theta_list = []
                range_list = []
                current_time = time
                theta_list.append(theta)
                range_list.append(range)

    return 0


def main():
    var = input("Data to be filtered: ")
    dataset = input("Dataset to be filtered: ")
    if (var == "multiple lidar"):
        print("FIltering lidar data to have multiple measurements per time step")
        filter_lidar_multiple(dataset)
    if (var == "lidar" or var == "encoder"):
        print("Filtering lidar and encoder data")
        filter_lidar_encoder(dataset)
    elif(var == "truth"):
        print("Filtering ground truth")
        filter_ground_truth()

    return 0


if __name__ == "__main__":
    main()
