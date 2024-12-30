import os
import numpy as np

import pandas as pd
from visual_kinematics.RobotSerial import *


def solve_aT3_6p(Link3TEnds, marker_points, threshold):

    ## solve aT3 = [r1, r2, r3, t1]     6^p = [p1, p2, p3]
    ##             [r4, r5, r6, t2]
    ##             [r7, r8, r9, t3]
    ##             [0, 0, 0, 1]
    ##  Formulate  Ax = b   A is 3n*15, x is 15*1, b is 3n*1
    ##    where x = [p1, p2, p3, r1, r2, r3, t1, r4, r5, r6, t2, r7, r8, r9, t3]^T

    assert len(marker_points) == len(Link3TEnds)
    n = len(marker_points)
    assert n >= 5

    A = np.zeros((3 * n, 15))
    b = np.zeros((3 * n, 1))

    for i in range(n):
        marker_point = [x / 1000.0 for x in marker_points[i]]

        Link3TEnd_i = Link3TEnds[i]

        ## [a1, a2, a3, b1]
        ## [a4, a5, a6, b2]
        ## [a7, a8, a9, b3]
        ## [0, 0, 0, 1]

        b[3 * i, 0] = -Link3TEnd_i[0][3]  # -b1
        b[3 * i + 1, 0] = -Link3TEnd_i[1][3]  # -b2
        b[3 * i + 2, 0] = -Link3TEnd_i[2][3]  # -b3

        A[3 * i][:3] = [Link3TEnd_i[0][0], Link3TEnd_i[0][1], Link3TEnd_i[0][2]]  # a1  a2  a3
        A[3 * i][3:6] = [-marker_point[0], -marker_point[1], -marker_point[2]]  # -q1, -q2, -q3
        A[3 * i][6] = -1

        A[3 * i + 1][:3] = [Link3TEnd_i[1][0], Link3TEnd_i[1][1], Link3TEnd_i[1][2]]  # a4  a5  a6
        A[3 * i + 1][7:10] = [-marker_point[0], -marker_point[1], -marker_point[2]]  # -q1, -q2, -q3
        A[3 * i + 1][10] = -1

        A[3 * i + 2][:3] = [Link3TEnd_i[2][0], Link3TEnd_i[2][1], Link3TEnd_i[2][2]]  # a7  a8  a9
        A[3 * i + 2][11:14] = [-marker_point[0], -marker_point[1], -marker_point[2]]  # -q1, -q2, -q3
        A[3 * i + 2][14] = -1

    x = np.linalg.lstsq(A, b, rcond=0)[0]

    error_all = A.dot(x) - b
    A_new = []
    b_new = []
    Link3TEnds_new = []
    marker_points_new = []
    for i in range(n):
        error_i = np.sqrt(
            np.square(error_all[3 * i]) + np.square(error_all[3 * i + 1]) + np.square(error_all[3 * i + 2]))
        if error_i < threshold:
            A_new.append(A[3 * i][:].tolist())
            A_new.append(A[3 * i + 1][:].tolist())
            A_new.append(A[3 * i + 2][:].tolist())
            b_new.append(b[3 * i])
            b_new.append(b[3 * i + 1])
            b_new.append(b[3 * i + 2])

            Link3TEnds_new.append(Link3TEnds[i])
            marker_points_new.append(marker_points[i])


    x = np.linalg.lstsq(A_new, b_new, rcond=0)[0]

    error_all = A.dot(x) - b
    A_new = []
    b_new = []
    Link3TEnds_new = []
    marker_points_new = []
    for i in range(n):
        error_i = np.sqrt(
            np.square(error_all[3 * i]) + np.square(error_all[3 * i + 1]) + np.square(error_all[3 * i + 2]))
        if error_i < threshold:
            A_new.append(A[3 * i][:].tolist())
            A_new.append(A[3 * i + 1][:].tolist())
            A_new.append(A[3 * i + 2][:].tolist())
            b_new.append(b[3 * i])
            b_new.append(b[3 * i + 1])
            b_new.append(b[3 * i + 2])

            Link3TEnds_new.append(Link3TEnds[i])
            marker_points_new.append(marker_points[i])

    # print(len(Link3TEnds_new))

    x = np.linalg.lstsq(A_new, b_new, rcond=0)[0]

    est_3Ta_ini = np.array([x[3:7, 0],
                            x[7:11, 0],
                            x[11:, 0],
                            [0, 0, 0, 1]])

    est_6p_ini = np.array([x[:3, 0]])

    est_3Ta_R_iter = est_3Ta_ini[:3, :3]
    # print(est_3Ta_R_iter)
    est_3Ta_R_iter = gram_schmidt(est_3Ta_R_iter)
    # print(est_3Ta_R_iter)
    est_3Ta_t_iter = est_3Ta_ini[:3, 3:]
    est_6p_iter = est_6p_ini

    for i in range(10000):
        est_3Ta_t_iter, est_6p_iter = solve_3Ta_t_6p(est_3Ta_R_iter, Link3TEnds_new, marker_points_new)
        est_3Ta_R_iter = solve_3Ta_R(est_3Ta_t_iter, est_6p_iter, Link3TEnds_new, marker_points_new)

    est_3Ta = np.identity(4)
    est_3Ta[:3, :3] = est_3Ta_R_iter
    est_3Ta[:3, 3] = est_3Ta_t_iter[:]
    est_6p = est_6p_iter

    return np.linalg.inv(est_3Ta), est_6p


def gram_schmidt(A):
    # Get the number of columns
    m, n = A.shape

    # Initialize an array for the orthogonalized vectors
    Q = np.zeros((m, n))

    for j in range(n):
        # Start with the original vector
        v = A[:, j]

        # Subtract the projections onto the previously computed orthogonal vectors
        for i in range(j):
            proj = np.dot(Q[:, i], v) / np.dot(Q[:, i], Q[:, i]) * Q[:, i]
            v -= proj

        # Normalize the vector
        Q[:, j] = v / np.linalg.norm(v)

    return Q


def solve_3Ta_t_6p(est_3Ta_R_iter, Link3TEnds, marker_points):
    num = len(Link3TEnds)

    A = np.zeros((3 * num, 6))
    b = np.zeros((3 * num, 1))

    for i in range(num):
        Link3TEnd_i = Link3TEnds[i]
        A[3 * i, :] = [Link3TEnd_i[0][0], Link3TEnd_i[0][1], Link3TEnd_i[0][2], -1.0, 0, 0]
        A[3 * i + 1, :] = [Link3TEnd_i[1][0], Link3TEnd_i[1][1], Link3TEnd_i[1][2], 0, -1.0, 0]
        A[3 * i + 2, :] = [Link3TEnd_i[2][0], Link3TEnd_i[2][1], Link3TEnd_i[2][2], 0, 0, -1.0]
        m = est_3Ta_R_iter.dot(np.array(marker_points[i])) / 1000.0
        n = np.array([Link3TEnd_i[0][3], Link3TEnd_i[1][3], Link3TEnd_i[2][3]])
        t = m - n
        b[3 * i, 0] = t[0]
        b[3 * i + 1, 0] = t[1]
        b[3 * i + 2, 0] = t[2]

    result = np.linalg.lstsq(A, b, rcond=0)[0]

    est_6p_iter = np.array(result[:3, 0])
    est_3Ta_t_iter = np.array(result[3:, 0])

    return est_3Ta_t_iter, est_6p_iter


def solve_3Ta_R(est_3Ta_t_iter, est_6p_iter, Link3TEnds, marker_points):
    num = len(Link3TEnds)

    A = np.zeros((num, 3))
    B = np.zeros((num, 3))

    for i in range(num):
        A[i, :] = np.array(marker_points[i]) / 1000.0
        Link3TEnd_i = np.array(Link3TEnds[i])
        B[i, :] = Link3TEnd_i[:3, :3].dot(est_6p_iter) + Link3TEnd_i[:3, 3] - est_3Ta_t_iter

    # Center the points
    centroid_a = np.mean(A, axis=0)
    centroid_b = np.mean(B, axis=0)

    centered_a = A - centroid_a
    centered_b = B - centroid_b

    # Compute the covariance matrix
    covariance_matrix = np.dot(A.T, B)

    # Singular Value Decomposition
    U, _, Vt = np.linalg.svd(covariance_matrix)

    # Compute the rotation matrix
    rotation_matrix = np.dot(Vt.T, U.T)

    # Ensure a right-handed coordinate system
    if np.linalg.det(rotation_matrix) < 0:
        Vt[2, :] *= -1
        rotation_matrix = np.dot(Vt.T, U.T)

    return rotation_matrix


if __name__ == "__main__":

    N_group = 6
    trackerdata_folder = "./calib_aT3_data/data_leika"
    robotdata_folder = "./calib_aT3_data/data_robot"
    group_lists = ["group1", "group2", "group3", "group4", "group5", "group6"]
    threshold = 0.00024                                                            ### maybe need to be adjusted

    # robot parameters
    dh_params = np.array([[0.1632, 0., 0.5 * pi, 0.],
                          [0., 0.647, pi, 0.5 * pi],
                          [0., 0.6005, pi, 0.],
                          [0.2013, 0., -0.5 * pi, -0.5 * pi],
                          [0.1025, 0., 0.5 * pi, 0.],
                          [0.094, 0., 0., 0.]])
    robot_aubo = RobotSerial(dh_params)

    # tracker data
    marker_points = []

    for group in group_lists:
        file_path = os.path.join(trackerdata_folder, group + ".csv")
        tracker_data = pd.read_csv(file_path, sep = ';')
        X = tracker_data['X  [mm]'].to_list()
        Y = tracker_data['Y  [mm]'].to_list()
        Z = tracker_data['Z  [mm]'].to_list()

        marker_points.extend([[X_i, Y_i, Z_i] for X_i, Y_i, Z_i in zip(X, Y, Z)])


    # robot data
    Link3TEnds = []
    for group in group_lists:
        file_path = os.path.join(robotdata_folder, group + ".txt")
        Link3TEnds_group = []

        with open(file_path, "r") as f:
            for line in f.readlines():
                if line != "\n":
                    line = line.strip('\n')

                    line_data = line.split(' ')
                    state = []
                    for data in line_data:
                        data = data.replace(",", "")
                        state.append(float(data))

                    f = robot_aubo.forward(np.array(state))

                    Ts = robot_aubo.ts
                    Link3TEnd_i = np.eye(4)
                    for j in range(3, 6):
                        Link3TEnd_i = Link3TEnd_i.dot(Ts[j].t_4_4)

                    Link3TEnds_group.append(Link3TEnd_i)

        Link3TEnds.extend(Link3TEnds_group)



    est_6p_aver = np.zeros((1, 3))
    est_aT3s = []
    for i in range(N_group):

        Link3TEnds_group = []
        marker_points_group = []
        # if i == 2:
        #     Link3TEnds_group = Link3TEnds[i*20:i*20 + 10]
        #     marker_points_group = marker_points[i * 20:i*20 + 10]
        # else:
        Link3TEnds_group = Link3TEnds[i * 20:(i + 1) * 20]
        marker_points_group = marker_points[i * 20:(i + 1) * 20]


        est_aT3, est_6p = solve_aT3_6p(Link3TEnds_group, marker_points_group, threshold)
        est_aT3s.append(est_aT3)

        print("Group", i, ":")
        print("est_aT3:", est_aT3)
        print("est_6p:", est_6p)
        print("\n")

        est_6p_aver = est_6p_aver + est_6p

    print("Average est_6p:", est_6p_aver/float(N_group))

    np.save('aT3s_init.npy', est_aT3s)
    np.save('p_init.npy', est_6p_aver/float(N_group))

