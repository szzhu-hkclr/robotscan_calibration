import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import open3d as o3d
import copy

from scipy.spatial.transform import Rotation as R
import pandas as pd
import math

from visual_kinematics.RobotSerial import *


def find_chessboard_corners(images, pattern_size, ShowCorners=False):
    """Finds the chessboard patterns and, if ShowImage is True, shows the images with the corners"""
    chessboard_corners = []
    IndexWithImg = []
    i = 0
    print("Finding corners...")
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, pattern_size)

        if ret:

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (10, 10), (-1, -1), criteria)

            chessboard_corners.append(corners)

            cv2.drawChessboardCorners(image, pattern_size, corners, ret)
            if ShowCorners:
                # plot image using maplotlib. The title should "Detected corner in image: " + i
                plt.imshow(image)
                plt.title("Detected corner in image: " + str(i))
                plt.show()
            # Save the image in a folder Named "DetectedCorners"
            # make folder
            # if not os.path.exists("DetectedCorners"):
            #     os.makedirs("DetectedCorners")

            # cv2.imwrite("DetectedCorners/DetectedCorners" + str(i) + ".png", image)

            IndexWithImg.append(i)
            i = i + 1
        else:
            print("No chessboard found in image: ", i)
            i = i + 1
    return chessboard_corners, IndexWithImg



def compute_camera_poses(chessboard_corners, pattern_size, square_size, intrinsic_matrix, dist, Testing=False):
    """Takes the chessboard corners and computes the camera poses"""
    # Create the object points.Object points are points in the real world that we want to find the pose of.
    object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    # Estimate the pose of the chessboard corners
    cam_T_chesss = []

    i = 1
    for corners in chessboard_corners:
        _, rvec, tvec = cv2.solvePnP(object_points, corners, intrinsic_matrix, dist)
        # rvec is the rotation vector, tvec is the translation vector
        if Testing == True:
            print("Current iteration: ", i, " out of ", len(chessboard_corners[0]), " iterations.")

            # Convert the rotation vector to a rotation matrix
            print("rvec: ", rvec)
            print("rvec[0]: ", rvec[0])
            print("rvec[1]: ", rvec[1])
            print("rvec[2]: ", rvec[2])
            print("--------------------")
        i = 1 + i
        R, _ = cv2.Rodrigues(rvec)  # R is the rotation matrix from the target frame to the camera frame

        cam_T_chess = np.eye(4)
        cam_T_chess[:3, :3] = R
        cam_T_chess[:3, 3] = tvec[:, 0]

        cam_T_chesss.append(cam_T_chess)

    return cam_T_chesss



# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z]) * 180 / math.pi


def get_angular_error(R_gt, R_est):
    """
    Get angular error
    """
    eu_gt = rotationMatrixToEulerAngles(R_gt)
    eu_est = rotationMatrixToEulerAngles(R_est)

    return np.average(np.abs(eu_gt - eu_est))

    # try:
    #     A = (np.trace(np.dot(R_gt.T, R_est)) - 1.0) / 2.0
    #     if A < -1:
    #         A = -1
    #
    #     if A > 1:
    #         A = 1
    #     rotError = math.fabs(math.acos(A))
    #     return math.degrees(rotError)
    # except ValueError:
    #     import pdb
    #     pdb.set_trace()
    #     return 99999

    # N = np.linalg.norm(R_gt - R_est, 'fro')
    # theta = 2 * np.arcsin(N/(2*np.sqrt(2)))
    # return theta*180.0/np.pi

def compute_transformation_diff(R_est, t_est, R_gt, t_gt):
    """
    Compute difference between two 4-by-4 SE3 transformation matrix
    """
    rot_error = get_angular_error(R_gt, R_est)
    trans_error = np.linalg.norm(t_gt - t_est)

    return rot_error, trans_error


if __name__ == '__main__':
    Setup603 = True
    if Setup603:
        square_size = 10.0 / 1000.0
        image_folder = "./pc_registration/2025-03-14_image_data"
        robotfile_path = os.path.join("./pc_registration/2025-03-14_robot_data.txt")
        intrinsic_matrix = np.array([[2.27516924e+03 ,0.00000000e+00, 1.41300356e+03],
                                    [0.00000000e+00 ,2.27392348e+03 ,8.94726558e+02],
                                    [0.00000000e+00 ,0.00000000e+00 ,1.00000000e+00]])

        dist =np.array( [[-1.21957894e-02 , 2.23164455e-02 ,-8.84981082e-06 , 2.15622170e-04,
  -2.16589555e-02]])
    else:
        # for photoneo
        square_size = 15.0 / 1000.0
        image_folder = "./pc_registration/2025-03-18_image_data"
        robotfile_path = os.path.join("./pc_registration/2025-03-18_robot_data .txt") 
        intrinsic_matrix = np.array([[2.25440219e+03, 0.00000000e+00, 1.01973189e+03],
                                     [0.00000000e+00, 2.25463546e+03, 7.81230173e+02],
                                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        dist = np.array([[-1.49250762e-01,  2.47190731e-01,  1.43050079e-04,  3.07944308e-05, -2.17900844e-01]])      



    ply_files = sorted(glob.glob(f'{image_folder}/*.ply'))

#------------------------------------------------------------------------------------------------------------
    ### robot_data
    robot_states = []
    with open(robotfile_path, "r") as f:
        for line in f.readlines():
            if line != "\n":
                line = line.strip('\n')

                line_data = line.split(' ')
                state = []
                for data in line_data:
                    data = data.replace(",", "")
                    state.append(float(data))
                robot_states.append(state)


    ###  6-DoF Kinematics
    if Setup603:
        dh_params = np.array([[0.1632, 0., 0.5 * pi, 0.],
                            [0., 0.647, pi, 0.5 * pi],
                            [0., 0.6005, pi, 0.],
                            [0.2013, 0., -0.5 * pi, -0.5 * pi],
                            [0.1025, 0., 0.5 * pi, 0.],
                            [0.094, 0., 0., 0.]])
        end_T_cam_ori=np.array([[ 0.9998491 , -0.0173042  , 0.00153237 , 0.04108286],
 [ 0.01729653 , 0.99983846 , 0.00488805 , 0.03589873],
 [-0.0016167 , -0.00486081 , 0.99998688 , 0.07195702],
 [ 0.     ,     0.      ,    0.   ,       1.        ]])
    else:
        dh_params = np.array([[0.550, 0.170, 0.5*pi, 0.0],
                          [0.0, 0.880, 0.0, 0.5*pi],
                          [0.0, 0.190, 0.5*pi, 0.0],
                          [0.810, 0.0, -0.5*pi, 0.0],
                          [0.00, 0.0, 0.5*pi, 0.0],               #
                          [0.115, 0.0, 0.0, 0.0]])
        # for photoneo setup in 16W
        # end_T_cam_ori = np.array([[-4.69116784e-04,  9.99961201e-01, -8.79638226e-03, -3.60227157e-02],
        #                     [-9.65813693e-01,  1.82728409e-03,  2.59230732e-01, -1.12813118e-01],
        #                     [ 2.59236747e-01,  8.61727592e-03,  9.65775363e-01,  1.20060987e-01],
        #                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        end_T_cam_ori = np.array([[-0.00100795,  0.99996224, -0.00863107, -0.03627546],
                                  [-0.96579093,  0.00126477,  0.25931886, -0.11289998],
                                  [ 0.25931998,  0.00859719,  0.9657532,   0.11992202],
                                  [ 0.,          0.,          0.,          1.        ]])

    robot_aubo_ori = RobotSerial(dh_params)

 

    base_T_camidxs = []
    for idx in range(len(robot_states)):
        robot_state = robot_states[idx]
        f = robot_aubo_ori.forward(np.array(robot_state))
        base_T_camidx = robot_aubo_ori.end_frame.t_4_4.dot(end_T_cam_ori)
        base_T_camidxs.append(base_T_camidx)

    base_T_cam1 = base_T_camidxs[0]
    K6DoF_cam1_T_camidxs = []
    for idx in range(len(base_T_camidxs) - 1):
        base_T_camidx = base_T_camidxs[idx + 1]
        K6DoF_cam1_T_camidx = np.linalg.inv(base_T_cam1).dot(base_T_camidx)

        ply_file = ply_files[idx + 1]
        pointcloud_2 = o3d.io.read_point_cloud(ply_file)
        K = K6DoF_cam1_T_camidx.copy()
        K[:3, 3] = 1000.0 * K[:3, 3]

        pointcloud_2_calib = copy.deepcopy(pointcloud_2).transform(K)
        new_ply_file = ply_file.replace(".ply", "_trans_6DoF.ply")
        o3d.io.write_point_cloud(new_ply_file, pointcloud_2_calib)

        K6DoF_cam1_T_camidxs.append(K6DoF_cam1_T_camidx)

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

    if Setup603:
        dh_params_new = np.array([[0.1632, 0., 0.5 * pi, 0.],
                            [0., 0.647, pi, 0.5 * pi],
                            [0., 0.6005, pi, 0.],
                            [2.0138808e-01 , 5.2008138e-05, -1.5709740e+00, -1.5708524e+00],
                            [1.0270937e-01 , 8.0966696e-05 , 1.5706815e+00 ,-1.9189264e-03],
                            [9.4042644e-02 ,-2.6792093e-05 ,-1.4009298e-04 , 2.6618787e-05]])
    else:
        dh_params_new = np.array([[0.550, 0.170, 0.5*pi, 0.0],
                            [0.0, 0.880, 0.0, 0.5*pi],
                            [0.0, 0.190, 0.5*pi, 0.0],
                            [8.10100675e-01, -3.19175801e-04, -1.57020402e+00,  1.19046235e-05],
                            [1.4248221e-04, 1.2206167e-04, 1.5704311e+00, 3.1969816e-04],
                            [1.1500100e-01,  4.0665665e-07, -4.1013078e-05,  1.0535180e-05]])
    robot_aubo_new = RobotSerial(dh_params_new)

    # tracker_T_3 poses
    if Setup603:
        posefile_path = "./2025-03-14_aT3s_opt.npy"
    else:
        posefile_path = "./aT3s_opt.npy"
    tracker_T_3s = np.load(posefile_path)
    if Setup603:
        end_T_cam =np.array([[ 0.99986931 ,-0.01605259 , 0.0019194 ,  0.04138638],
 [ 0.01604281 , 0.99985878 , 0.00500476 , 0.03565126],
 [-0.00199947 ,-0.00497332 , 0.99998563 , 0.0714801 ],
 [ 0.  ,        0.     ,     0.     ,     1.        ]])
    else:
        # for 16W photoneo setup
        end_T_cam = np.array([[-4.69196201e-04,  9.99961201e-01, -8.79633495e-03, -3.60227528e-02],
                              [-9.65813707e-01,  1.82719465e-03,  2.59230678e-01, -1.12813121e-01],
                              [ 2.59236693e-01,  8.61725091e-03,  9.65775378e-01,  1.20060964e-01],
                              [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    tracker_T_camidxs = []
    for idx in range(len(robot_states)):
        robot_state = robot_states[idx]
        f = robot_aubo_new.forward(np.array(robot_state))

        Ts = robot_aubo_new.ts
        Link3TEnd_i = np.eye(4)
        for j in range(3, 6):
            Link3TEnd_i = Link3TEnd_i.dot(Ts[j].t_4_4)

        if idx<2:
            tracker_T_3s_temp=tracker_T_3s[0]
        if 2<=idx<3:
            tracker_T_3s_temp=tracker_T_3s[1]
        if 3<=idx<5:
            tracker_T_3s_temp=tracker_T_3s[2]
        if 5<=idx:
            tracker_T_3s_temp=tracker_T_3s[4]
        
        tracker_T_camidx = tracker_T_3s_temp.dot(Link3TEnd_i).dot(end_T_cam)

        tracker_T_camidxs.append(tracker_T_camidx)

    tracker_T_cam1 = tracker_T_camidxs[0]
    K3DoF_cam1_T_camidxs = []
    for idx in range(len(tracker_T_camidxs) - 1):
        tracker_T_camidx = tracker_T_camidxs[idx + 1]
        K3DoF_cam1_T_camidx = np.linalg.inv(tracker_T_cam1).dot(tracker_T_camidx)

        ply_file = ply_files[idx + 1]
        pointcloud_2 = o3d.io.read_point_cloud(ply_file)
        K = K3DoF_cam1_T_camidx.copy()
        K[:3, 3] = 1000.0 * K[:3, 3]

        pointcloud_2_calib = copy.deepcopy(pointcloud_2).transform(K)
        new_ply_file = ply_file.replace(".ply", "_trans_3DoF.ply")
        o3d.io.write_point_cloud(new_ply_file, pointcloud_2_calib)

        K3DoF_cam1_T_camidxs.append(K3DoF_cam1_T_camidx)

