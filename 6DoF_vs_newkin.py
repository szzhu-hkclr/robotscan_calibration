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
    Setup603 = False

    if Setup603:
        image_folder = "./evaluate_data/603_data/image_data"
        robotfile_path = os.path.join("./evaluate_data/603_data/robot_data.txt")
    else:
        image_folder = "./evaluate_data/16w_data/image_data"
        robotfile_path = os.path.join("./evaluate_data/16w_data/robot_data.txt")

    images = []
    image_files = sorted(glob.glob(f'{image_folder}/*_2DImage.png'))
    images_group = [cv2.imread(f) for f in image_files]
    images.extend(images_group)

    ply_files = sorted(glob.glob(f'{image_folder}/*_textured.ply'))

    pattern_size = (11, 8)
    if Setup603:
        square_size = 10 / 1000
    else:    
        square_size = 15 / 1000

    ShowProjectError = True
    ShowCorners = False

    # camera calibration
    chessboard_corners, IndexWithImg = find_chessboard_corners(images, pattern_size, ShowCorners=ShowCorners)

    if Setup603:
        # mecheye nano ultra
        intrinsic_matrix = np.array([[2.28169027e+03, 0.00000000e+00, 1.40720528e+03],
                                    [0.00000000e+00, 2.28099962e+03, 8.93295531e+02],
                                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    else:
        # photoneo model s
        intrinsic_matrix = np.array([[2.28169027e+03, 0.00000000e+00, 1.40720528e+03],
                                     [0.00000000e+00, 2.28099962e+03, 8.93295531e+02],
                                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


    dist = np.array([[-7.14967118e-03, -1.69424563e-03, -4.70604299e-05, -2.47795750e-04, 4.36516991e-02]])

    cam_T_chesss = compute_camera_poses(chessboard_corners, pattern_size, square_size, intrinsic_matrix, dist)

    cam1_T_chess = cam_T_chesss[0]

    cam_cam1_T_camidxs = []
    for idx in range(len(cam_T_chesss) - 1):
        camidx_T_chess = cam_T_chesss[idx + 1]
        cam1_T_camidx = cam1_T_chess.dot(np.linalg.inv(camidx_T_chess))

        ply_file = ply_files[idx + 1]
        pointcloud_2 = o3d.io.read_point_cloud(ply_file)
        K = cam1_T_camidx.copy()
        if Setup603:
            # scales the translation vector from meters to millimeters
            K[:3, 3] = 1000.0 * K[:3, 3]
        else:
            # No scaling needed when point cloud is in meters
            K[:3, 3] = K[:3, 3]
        pointcloud_2_calib = copy.deepcopy(pointcloud_2).transform(K)
        new_ply_file = ply_file.replace(".ply", "_trans_cam.ply")
        o3d.io.write_point_cloud(new_ply_file, pointcloud_2_calib)

        cam_cam1_T_camidxs.append(cam1_T_camidx)


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
        # aubo i10
        dh_params = np.array([[0.1632, 0., 0.5 * pi, 0.],
                            [0., 0.647, pi, 0.5 * pi],
                            [0., 0.6005, pi, 0.],
                            [0.2013, 0., -0.5 * pi, -0.5 * pi],
                            [0.1025, 0., 0.5 * pi, 0.],
                            [0.094, 0., 0., 0.]])
        end_T_cam_ori = np.array([[9.99994962e-01, 5.77411071e-04, -3.12120952e-03, 0.04256417],
                                [-5.59361481e-04, 9.99983135e-01, 5.78066586e-03, 0.00681424],
                                [3.12449470e-03, -5.77889085e-03, 9.99978421e-01, 0.12986917],
                                [0, 0, 0, 1]])
    else:
        # nachi mz25
                                # |  d  |  a  |  alpha  |  theta  |
        dh_params = np.array([
                                [ 0.2495,  0,      0.5 * pi,  0],        # Joint 1
                                [ 0.3005,  0.17,   0,         0.5 * pi], # Joint 2
                                [ 0.88,    0.157,  0.5 * pi,  0],        # Joint 3
                                [ 0.19,    0.81,  -0.5 * pi,  0],        # Joint 4
                                [ 0,       0,      0.5 * pi,  0],        # Joint 5
                                [ 0,       0.101,  0,         0]         # Joint 6
                             ])
        end_T_cam_ori = np.array([[9.99994962e-01, 5.77411071e-04, -3.12120952e-03, 0.04256417],
                                 [-5.59361481e-04, 9.99983135e-01, 5.78066586e-03, 0.00681424],
                                 [3.12449470e-03, -5.77889085e-03, 9.99978421e-01, 0.12986917],
                                 [0, 0, 0, 1]])
    robot_ori = RobotSerial(dh_params)
    
    base_T_camidxs = []
    for idx in range(len(robot_states)):
        robot_state = robot_states[idx]
        f = robot_ori.forward(np.array(robot_state))
        base_T_camidx = robot_ori.end_frame.t_4_4.dot(end_T_cam_ori)
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


    dh_params_new = np.array([[0.1632, 0., 0.5 * pi, 0.],
                          [0., 0.647, pi, 0.5 * pi],
                          [0., 0.6005, pi, 0.],
                          [2.0132992e-01, 1.5028477e-05, -1.5706592e+00, -1.5707269e+00],
                          [1.02672115e-01, 4.72186694e-05, 1.57062984e+00, -2.43247976e-03],
                          [9.4024345e-02, 8.5766565e-05, -7.1511102e-05, -8.5101266e-05]])
    robot_new = RobotSerial(dh_params_new)

    # tracker_T_3 poses
    posefile_path = "./aT3s_opt.npy"
    tracker_T_3s = np.load(posefile_path)

    end_T_cam = np.array([[0.99999378, 0.00218076, -0.00277091, 0.04262605],
                         [-0.00216497, 0.99998148, 0.00568824, 0.0067556],
                         [0.00278326, -0.00568221, 0.99997998, 0.12970971],
                         [0, 0, 0, 1]])

    tracker_T_camidxs = []
    for idx in range(len(robot_states)):
        robot_state = robot_states[idx]
        f = robot_new.forward(np.array(robot_state))

        Ts = robot_new.ts
        Link3TEnd_i = np.eye(4)
        for j in range(3, 6):
            Link3TEnd_i = Link3TEnd_i.dot(Ts[j].t_4_4)

        tracker_T_camidx = tracker_T_3s[idx].dot(Link3TEnd_i).dot(end_T_cam)

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

    errors_R_K6DoF = []
    errors_t_K6DoF = []
    errors_R_K3DoF = []
    errors_t_K3DoF = []
    for i in range(len(cam_cam1_T_camidxs)):
        cam_cam1_T_cami = cam_cam1_T_camidxs[i]
        K6DoF_cam1_T_cami = K6DoF_cam1_T_camidxs[i]
        K3DoF_cam1_T_cami = K3DoF_cam1_T_camidxs[i]

        error_R_K6DoF, error_t_K6DoF = compute_transformation_diff(K6DoF_cam1_T_cami[:3, :3], K6DoF_cam1_T_cami[:3, 3:], cam_cam1_T_cami[:3, :3], cam_cam1_T_cami[:3, 3:])
        error_R_K3DoF, error_t_K3DoF = compute_transformation_diff(K3DoF_cam1_T_cami[:3, :3], K3DoF_cam1_T_cami[:3, 3:], cam_cam1_T_cami[:3, :3], cam_cam1_T_cami[:3, 3:])

        errors_R_K6DoF.append(error_R_K6DoF)
        errors_R_K3DoF.append(error_R_K3DoF)
        if Setup603:
            errors_t_K6DoF.append(error_t_K6DoF * 1000)
            errors_t_K3DoF.append(error_t_K3DoF * 1000)
        else:
            # No scaling needed when point cloud is in meters
            errors_t_K6DoF.append(error_t_K6DoF)
            errors_t_K3DoF.append(error_t_K3DoF)

    plt.subplot(2, 1, 1)

    plt.plot(range(len(errors_R_K6DoF)), errors_R_K6DoF, label="6DOF_R_error")
    plt.plot(range(len(errors_R_K3DoF)), errors_R_K3DoF, label="Ours_R_error")
    plt.title("Rotation Error")
    plt.ylim(0, 0.2)

    plt.xlabel('Index')
    plt.ylabel('Error (deg.)')
    plt.legend()

    plt.subplot(2, 1, 2)

    plt.plot(range(len(errors_t_K6DoF)), errors_t_K6DoF, label="6DOF_t_error")
    plt.plot(range(len(errors_t_K3DoF)), errors_t_K3DoF, label="Ours_t_error")
    plt.title("Translation Error")
    if Setup603:
        plt.ylim(0, 3)
    else:
        # update the ylim to a range that reflects meter-scale errors
        plt.ylim(0, 0.003)

    plt.xlabel('Index')
    plt.ylabel('Error (mm)')

    plt.subplots_adjust(hspace=0.6)

    plt.legend()
    plt.show()


    print("errors_R_K6DoF", errors_R_K6DoF)
    print("errors_t_K6DoF", errors_t_K6DoF)
    print("errors_R_K3DoF", errors_R_K3DoF)
    print("errors_t_K3DoF", errors_t_K3DoF)