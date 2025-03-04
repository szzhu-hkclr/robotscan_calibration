import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os

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


def calculate_intrinsics(chessboard_corners, IndexWithImg, pattern_size, square_size, ImgSize, ShowProjectError=False):
    """Calculates the intrinc camera parameters fx, fy, cx, cy from the images"""
    # Find the corners of the chessboard in the image
    imgpoints = chessboard_corners
    # Find the corners of the chessboard in the real world
    objpoints = []
    for i in range(len(IndexWithImg)):
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
        objpoints.append(objp)
    # Find the intrinsic matrix
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, ImgSize, None, None)

    print("The projection error from the calibration is: ",
          calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist, ShowProjectError))
    return mtx, dist


def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist, ShowPlot=False):
    """Calculates the reprojection error of the camera for each image. The output is the mean reprojection error
    If ShowPlot is True, it will show the reprojection error for each image in a bar graph"""

    total_error = 0
    num_points = 0
    errors = []

    for i in range(len(objpoints)):
        imgpoints_projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        imgpoints_projected = imgpoints_projected.reshape(-1, 1, 2)
        error = cv2.norm(imgpoints[i], imgpoints_projected, cv2.NORM_L2) / len(imgpoints_projected)
        errors.append(error)
        total_error += error
        num_points += 1

    mean_error = total_error / num_points

    # if ShowPlot:
    #     # Plotting the bar graph
    #     fig, ax = plt.subplots()
    #     img_indices = range(1, len(errors) + 1)
    #     ax.bar(img_indices, errors)
    #     ax.set_xlabel('Image Index')
    #     ax.set_ylabel('Reprojection Error')
    #     ax.set_title('Reprojection Error for Each Image')
    #     plt.show()
    #     print(errors)

    #     #Save the bar plot as a .png
    #     fig.savefig('ReprojectionError.png')

    return mean_error


def compute_camera_poses(chessboard_corners, pattern_size, square_size, intrinsic_matrix, dist, Testing=False):
    """Takes the chessboard corners and computes the camera poses"""
    # Create the object points.Object points are points in the real world that we want to find the pose of.
    object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    # Estimate the pose of the chessboard corners
    RTarget2Cam = []
    TTarget2Cam = []
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
        RTarget2Cam.append(R)
        TTarget2Cam.append(tvec)

    return RTarget2Cam, TTarget2Cam


def read_joints(file):
    endposes = []
    with open(file, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            # line = line.replace(" ", "")

            line_data = line.split(' ')
            point = []
            for data in line_data:
                data = data.replace(",", "")
                point.append(float(data))

            endposes.append(point)

    return endposes

Setup603 = False
SetupMechEye = False
SetupPatternSize6 = True
group_lists = []

if Setup603:
    image_folder = "./handeye_data/data_images"
    pose_folder = "./handeye_data/data_robot"
    group_lists = ["group1", "group2", "group3", "group4", "group5", "group6"]
    square_size = 10 / 1000
else:
    if SetupMechEye:
        image_folder = "./handeye_data/data_mecheye"
    else:
        image_folder = "./handeye_data/data_photoneo"
    if SetupPatternSize6:
        square_size = 6 / 1000
    else:
        square_size = 15 / 1000

    pose_folder = "./handeye_data/data_nachi"
    group_lists = ["group1", "group2", "group3", "group4", "group5", "group6"]

pattern_size = (11, 8)

ShowProjectError = True
ShowCorners = False


images = []
for group in group_lists:
    image_files = sorted(glob.glob(f'{image_folder}/{group}/*.png'))
    images_group = [cv2.imread(f) for f in image_files]
    images.extend(images_group)

# camera calibration
chessboard_corners, IndexWithImg = find_chessboard_corners(images, pattern_size, ShowCorners=ShowCorners)
intrinsic_matrix, dist = calculate_intrinsics(chessboard_corners, IndexWithImg,
                                        pattern_size, square_size,
                                        images[0].shape[:2][::-1], ShowProjectError=ShowProjectError)



# Calculate camera extrinsics
RTarget2Cam, TTarget2Cam = compute_camera_poses(chessboard_corners, pattern_size, square_size, intrinsic_matrix, dist)


REnd2Base = []
TEnd2Base = []
if Setup603:
    # aubo i10
    dh_params = np.array([[0.1632, 0., 0.5 * pi, 0.],
                        [0., 0.647, pi, 0.5 * pi],
                        [0., 0.6005, pi, 0.],
                        [0.2013, 0., -0.5 * pi, -0.5 * pi],
                        [0.1025, 0., 0.5 * pi, 0.],
                        [0.094, 0., 0., 0.]])
else:
    # nachi mz25
    # by yz:
    #   |  d     |  a      | alpha  | theta|
    dh_params = np.array([
        [0.55,    0.17,     pi / 2,      0.],          
        [0.,      0.88,     0.,      pi / 2],     
        [0.,      0.19,     pi / 2,      0.],    
        [0.81, 0.0, -1.57079632679, 0.0],
        [0.0, 0.0, 1.57079632679, 0.0],
        [0.115, 0.0, 0.0, 0.0]
    ])
robot = RobotSerial(dh_params)

img_counter = 0
valid_img_counter = 0
for group in group_lists:
    pose_file = f'{pose_folder}/{group}.txt'
    pose_group = read_joints(pose_file)

    for pose in pose_group:
        # Check if this image had a successful chessboard detection
        if img_counter in IndexWithImg:
            f = robot.forward(np.array(pose))
            T = robot.end_frame.t_4_4
            REnd2Base.append(T[:3, :3])
            TEnd2Base.append(T[:3, 3])
            valid_img_counter += 1
        img_counter += 1

REnd2Base = np.array(REnd2Base)
TEnd2Base = np.array(TEnd2Base)

# Now the sizes should match
print(f"Applied images: {valid_img_counter}/{img_counter}")
print(f"Number of valid robot poses: {len(REnd2Base)}")
print(f"Number of valid camera poses: {len(RTarget2Cam)}")



R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    REnd2Base,
    TEnd2Base,
    RTarget2Cam,
    TTarget2Cam,
    method=4
)
end_T_cam = np.eye(4)
end_T_cam[:3, :3] = R_cam2gripper
end_T_cam[:3, 3] = t_cam2gripper[:, 0]

print("end_T_cam: ", end_T_cam)
print("Intrinsics: ", intrinsic_matrix)
print("Distortion: ", dist)

# end_T_cam = np.array([[9.99994962e-01, 5.77411071e-04, -3.12120952e-03, 0.04256417],
#                      [-5.59361481e-04, 9.99983135e-01, 5.78066586e-03, 0.00681424],
#                      [3.12449470e-03, -5.77889085e-03, 9.99978421e-01, 0.12986917],
#                      [0, 0, 0, 1]])
#
# intrinsic_matrix = np.array([[2.28169027e+03, 0.00000000e+00, 1.40720528e+03],
#                              [0.00000000e+00, 2.28099962e+03, 8.93295531e+02],
#                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
#
# dist = np.array([[-7.14967118e-03, -1.69424563e-03, -4.70604299e-05, -2.47795750e-04, 4.36516991e-02]])

# 16w original
# end_T_cam:  [[ 1.01119879e-04  9.99959590e-01 -8.98933208e-03 -3.59683578e-02]
#  [-9.65835028e-01  2.42731540e-03  2.59146305e-01 -1.12898917e-01]
#  [ 2.59157653e-01  8.65600696e-03  9.65796244e-01  1.19833855e-01]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
# Intrinsics:  [[2.25440219e+03 0.00000000e+00 1.01973189e+03]
#  [0.00000000e+00 2.25463546e+03 7.81230173e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
# Distortion:  [[-1.49250762e-01  2.47190731e-01  1.43050079e-04  3.07944308e-05
#   -2.17900844e-01]]

# 16w 2025-02-04
# end_T_cam:  [[ 1.93479737e-04  9.99962018e-01 -8.71352487e-03 -3.60409880e-02]
#  [-9.65509284e-01  2.45552737e-03  2.60357049e-01 -1.12627550e-01]
#  [ 2.60368556e-01  8.36261535e-03  9.65473087e-01  1.19650637e-01]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
# Intrinsics:  [[2.25305711e+03 0.00000000e+00 1.01559570e+03]
#  [0.00000000e+00 2.25360654e+03 7.81779803e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
# Distortion:  [[-1.54898718e-01  3.31404747e-01  1.84773680e-04 -2.48342547e-04
#   -5.43549319e-01]]

# 2025-02-05 mecheye
# end_T_cam:  [[ 6.57561539e-03  9.99929240e-01  9.91339433e-03  7.35224848e-02]
#  [-9.99978347e-01  6.57784286e-03 -1.92103993e-04 -4.20281285e-02]
#  [-2.57299149e-04 -9.91191648e-03  9.99950843e-01  7.60294999e-02]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
# Intrinsics:  [[2.28057878e+03 0.00000000e+00 1.40806928e+03]
#  [0.00000000e+00 2.28009032e+03 8.93363155e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
# Distortion:  [[-0.00242916  0.02332046 -0.00018823  0.00046113 -0.0762033 ]]

# 2025-02-24 photoneo
# end_T_cam:  [[-2.54702105e-02 -9.99675240e-01 -8.26100925e-04  3.88897603e-02]
#  [ 9.61194401e-01 -2.42626884e-02 -2.74802920e-01  1.17314504e-01]
#  [ 2.74693631e-01 -7.79333180e-03  9.61500220e-01  1.27352855e-01]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
# Intrinsics:  [[2.24844948e+03 0.00000000e+00 1.02513990e+03]
#  [0.00000000e+00 2.24714713e+03 7.80309743e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
# Distortion:  [[-0.15319278  0.29214735 -0.00036029  0.00047065 -0.34778661]]