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
    SetupMechEye = False
    OnlyGroup3n4 = False
    
    # handle arbitrary group IDs when tracker and robot data are not aligned
    robot_states_group_lists = ["group1", "group2", "group3", "group4", "group5", "group6"]
    image_group_lists = ["group1", "group2", "group3", "group4", "group5", "group6"]
    tracker_group_lists = ["group1", "group2", "group3", "group4", "group5", "group6"]

    if Setup603:
        square_size = 10.0 / 1000.0
        pattern_size = (11, 8)
        image_folder = "./evaluate_data/603_data/image_data"
        robotfile_path = os.path.join("./evaluate_data/603_data/robot_data.txt")
        intrinsic_matrix = np.array([[2.28169027e+03, 0.00000000e+00, 1.40720528e+03],
                                    [0.00000000e+00, 2.28099962e+03, 8.93295531e+02],
                                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        dist = np.array([[-7.14967118e-03, -1.69424563e-03, -4.70604299e-05, -2.47795750e-04, 4.36516991e-02]])
    else:
        if OnlyGroup3n4:
            square_size = 6 / 1000
        else:
            square_size = 15 / 1000
        pattern_size = (11, 8)
        image_folder = "./evaluate_data/16w_data/image_data"
        robotfile_path = os.path.join("./evaluate_data/16w_data/robot_data.txt")
        if SetupMechEye:
            # 2025-02-05 mecheye
            intrinsic_matrix = np.array([[2.28057878e+03, 0.00000000e+00, 1.40806928e+03],
                                        [0.00000000e+00, 2.28009032e+03, 8.93363155e+02],
                                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
            dist = np.array([[-0.00242916,  0.02332046, -0.00018823,  0.00046113, -0.0762033]])        
        else:
            
            # cam readings
            # intrinsic_matrix = np.array([[2247.24, 0., 1019.52],
            #                              [0., 2245.15, 781.512],
            #                              [0., 0., 1.]])
            # dist = np.array([[-0.127476, 0.157168, 0.000106415, -0.000806452, -0.04221411]]) 

            if OnlyGroup3n4:
                # 2025-02-24
                intrinsic_matrix = np.array([[2.24844948e+03, 0.00000000e+00, 1.02513990e+03],
                                             [0.00000000e+00, 2.24714713e+03, 7.80309743e+02],
                                             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
                dist = np.array([[-0.15319278,  0.29214735, -0.00036029,  0.00047065, -0.34778661]])     
                tracker_group_lists = ["group3", "group4"]  # Only have data for groups 3 and 4   
            else:
                # 16w 2025-01-17
                # intrinsic_matrix = np.array([[2.25440219e+03, 0.00000000e+00, 1.01973189e+03],
                #                             [0.00000000e+00, 2.25463546e+03, 7.81230173e+02],
                #                             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
                # dist = np.array([[-1.49250762e-01,  2.47190731e-01,  1.43050079e-04,  3.07944308e-05, -2.17900844e-01]])      
                # 16w 2025-02-04
                intrinsic_matrix = np.array([[2.25305711e+03, 0.00000000e+00, 1.01559570e+03],
                                            [0.00000000e+00, 2.25360654e+03, 7.81779803e+02],
                                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
                dist = np.array([[-1.54898718e-01, 3.31404747e-01, 1.84773680e-04, -2.48342547e-04, -5.43549319e-01]])   
                # 16w 2025-02-24     
                # intrinsic_matrix = np.array([[2.24844931e+03, 0.00000000e+00, 1.02513974e+03],
                #                              [0.00000000e+00, 2.24714698e+03, 7.80309845e+02],
                #                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
                # dist = np.array([[-0.15319236,  0.29214266, -0.00036031,  0.00047066, -0.34777189]])   
    
    # Create mapping between all three datasets
    # Find the groups that exist in all three lists
    valid_groups = [group for group in robot_states_group_lists 
                   if group in image_group_lists and group in tracker_group_lists]
    
    # Create mappings from group name to respective indices in each list
    robot_indices = {group: robot_states_group_lists.index(group) for group in valid_groups}
    image_indices = {group: image_group_lists.index(group) for group in valid_groups}
    tracker_indices = {group: tracker_group_lists.index(group) for group in valid_groups}

    print(f"Valid groups: {valid_groups}")
    print(f"Processing only {len(valid_groups)} states with valid data across all three sources")

    # Load images based on valid groups
    images = []
    image_files = sorted(glob.glob(f'{image_folder}/*_2DImage.png'))
    
    # Filter the images to only include those from valid groups
    valid_images = []
    valid_image_indices = []
    for group in valid_groups:
        img_idx = image_indices[group]
        if img_idx < len(image_files):  # Safety check
            valid_images.append(cv2.imread(image_files[img_idx]))
            valid_image_indices.append(img_idx)
    
    images.extend(valid_images)

    # Get ply files from valid groups
    ply_files = sorted(glob.glob(f'{image_folder}/*_textured.ply'))
    valid_ply_files = [ply_files[idx] for idx in valid_image_indices if idx < len(ply_files)]

    ShowProjectError = True
    ShowCorners = False

    # Camera calibration
    chessboard_corners, IndexWithImg = find_chessboard_corners(images, pattern_size, ShowCorners=ShowCorners)

    # Only continue if corners were detected
    if len(chessboard_corners) == 0:
        print("No chessboard corners detected in any images. Cannot proceed.")
        exit(1)

    cam_T_chesss = compute_camera_poses(chessboard_corners, pattern_size, square_size, intrinsic_matrix, dist)

    cam1_T_chess = cam_T_chesss[0]

    cam_cam1_T_camidxs = []
    for idx in range(len(cam_T_chesss) - 1):
        camidx_T_chess = cam_T_chesss[idx + 1]
        cam1_T_camidx = cam1_T_chess.dot(np.linalg.inv(camidx_T_chess))

        if idx < len(valid_ply_files) - 1:  # Make sure we have a valid ply file
            ply_file = valid_ply_files[idx + 1]
            pointcloud_2 = o3d.io.read_point_cloud(ply_file)
            K = cam1_T_camidx.copy()
            if Setup603 or SetupMechEye:
                K[:3, 3] = 1000.0 * K[:3, 3]
            else:  # No scaling needed when point cloud is in meters
                K[:3, 3] = K[:3, 3]
            pointcloud_2_calib = copy.deepcopy(pointcloud_2).transform(K)
            new_ply_file = ply_file.replace(".ply", "_trans_cam.ply")
            o3d.io.write_point_cloud(new_ply_file, pointcloud_2_calib)

        cam_cam1_T_camidxs.append(cam1_T_camidx)

    ### Load robot data
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

    # Filter robot states to only include valid groups
    valid_robot_states = [robot_states[robot_indices[group]] for group in valid_groups 
                          if robot_indices[group] < len(robot_states)]

    ###  6-DoF Kinematics
    if Setup603:
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
        dh_params = np.array([[0.550, 0.170, 0.5*pi, 0.0],
                          [0.0, 0.880, 0.0, 0.5*pi],
                          [0.0, 0.190, 0.5*pi, 0.0],
                          [0.810, 0.0, -0.5*pi, 0.0],
                          [0.00, 0.0, 0.5*pi, 0.0],               #
                          [0.115, 0.0, 0.0, 0.0]])
        if SetupMechEye:
            # 2025-02-05 mecheye
            end_T_cam_ori = np.array([[ 6.57561539e-03,  9.99929240e-01,  9.91339433e-03,  7.35224848e-02],
                                      [-9.99978347e-01,  6.57784286e-03, -1.92103993e-04, -4.20281285e-02],
                                      [-2.57299149e-04, -9.91191648e-03,  9.99950843e-01,  7.60294999e-02],
                                      [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        else:
            if OnlyGroup3n4:
                # 16w 02-24
                end_T_cam_ori = np.array([[-2.54702105e-02, -9.99675240e-01, -8.26100925e-04,  3.88897603e-02],
                                          [ 9.61194401e-01, -2.42626884e-02, -2.74802920e-01,  1.17314504e-01],
                                          [ 2.74693631e-01, -7.79333180e-03,  9.61500220e-01,  1.27352855e-01],
                                          [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
            else:
                # 16w 01-17
                # end_T_cam_ori = np.array([[-0.00100795,  0.99996224, -0.00863107, -0.03627546],
                #                           [-0.96579093,  0.00126477,  0.25931886, -0.11289998],
                #                           [ 0.25931998,  0.00859719,  0.9657532,   0.11992202],
                #                           [ 0.,          0.,          0.,          1.        ]])
                # 16w 02-04
                end_T_cam_ori = np.array([[-3.87624226e-04,  9.99963614e-01, -8.52170583e-03, -3.61326405e-02],
                                          [-9.65486865e-01,  1.84525852e-03,  2.60445211e-01, -1.12564643e-01],
                                          [ 2.60451459e-01,  8.32854992e-03,  9.65451020e-01,  1.19847006e-01],
                                          [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
                # 16w 02-24
                # end_T_cam_ori = np.array([[-0.04423496, -0.99901788,  0.00255733,  0.05719259],
                #                           [ 0.95842326, -0.04315954, -0.28206755,  0.27836189],
                #                           [ 0.2819009,  -0.01002624,  0.95939114, -0.53584164],
                #                           [ 0.,          0.,          0.,          1.        ]])
    
    robot_ori = RobotSerial(dh_params)
    
    # Process only valid robot states
    base_T_camidxs = []
    for robot_state in valid_robot_states:
        f = robot_ori.forward(np.array(robot_state))
        base_T_camidx = robot_ori.end_frame.t_4_4.dot(end_T_cam_ori)
        base_T_camidxs.append(base_T_camidx)

    if not base_T_camidxs:
        print("No valid robot states found. Cannot proceed.")
        exit(1)

    base_T_cam1 = base_T_camidxs[0]
    K6DoF_cam1_T_camidxs = []
    
    for idx in range(len(base_T_camidxs) - 1):
        base_T_camidx = base_T_camidxs[idx + 1]
        K6DoF_cam1_T_camidx = np.linalg.inv(base_T_cam1).dot(base_T_camidx)

        if idx + 1 < len(valid_ply_files):  # Make sure we have a valid ply file
            ply_file = valid_ply_files[idx + 1]
            pointcloud_2 = o3d.io.read_point_cloud(ply_file)
            K = K6DoF_cam1_T_camidx.copy()
            if Setup603 or SetupMechEye:
                K[:3, 3] = 1000.0 * K[:3, 3]
            else:  # No scaling needed when point cloud is in meters
                K[:3, 3] = K[:3, 3]
            pointcloud_2_calib = copy.deepcopy(pointcloud_2).transform(K)
            new_ply_file = ply_file.replace(".ply", "_trans_6DoF.ply")
            o3d.io.write_point_cloud(new_ply_file, pointcloud_2_calib)

        K6DoF_cam1_T_camidxs.append(K6DoF_cam1_T_camidx)

    if Setup603:
        dh_params_new = np.array([[0.1632, 0., 0.5 * pi, 0.],
                            [0., 0.647, pi, 0.5 * pi],
                            [0., 0.6005, pi, 0.],
                            [2.0132992e-01, 1.5028477e-05, -1.5706592e+00, -1.5707269e+00],
                            [1.02672115e-01, 4.72186694e-05, 1.57062984e+00, -2.43247976e-03],
                            [9.4024345e-02, 8.5766565e-05, -7.1511102e-05, -8.5101266e-05]])
    else:
        if OnlyGroup3n4:
            dh_params_new = np.array([[0.550, 0.170, 0.5*pi, 0.0],
                            [0.0, 0.880, 0.0, 0.5*pi],
                            [0.0, 0.190, 0.5*pi, 0.0],
                            [ 8.0983901e-01,  4.0477642e-04, -1.5709044e+00, -5.5365084e-04],
                            [-3.2921496e-04, -2.9793917e-04,  1.5709493e+00,  1.2638936e-03],
                            [ 1.1520112e-01,  5.2975509e-05, -1.1862504e-04, -6.4109037e-05]])
        else:
            # 16w 01-17
            dh_params_new = np.array([[0.550, 0.170, 0.5*pi, 0.0],
                            [0.0, 0.880, 0.0, 0.5*pi],
                            [0.0, 0.190, 0.5*pi, 0.0],
                            [8.10100675e-01, -3.19175801e-04, -1.57020402e+00,  1.19046235e-05],
                            [1.4248221e-04, 1.2206167e-04, 1.5704311e+00, 3.1969816e-04],
                            [1.1500100e-01,  4.0665665e-07, -4.1013078e-05,  1.0535180e-05]])
            # 16w 02-24
            # dh_params_new = np.array([[0.550, 0.170, 0.5*pi, 0.0],
            #                 [0.0, 0.880, 0.0, 0.5*pi],
            #                 [0.0, 0.190, 0.5*pi, 0.0],
            #                 [4.1023746e-01, 8.8334800e-06,-1.5701793e+00, -3.6477974e-01],
            #                 [-2.0575602e-05,-1.7802379e-04, 1.5701313e+00, 1.0053637e-03],
            #                 [0.26512548,     0.04879135,    0.01211041,    0.04397908]])
    robot_new = RobotSerial(dh_params_new)

    # tracker_T_3 poses
    posefile_path = "./aT3s_opt.npy"
    try:
        tracker_T_3s = np.load(posefile_path)
        print(f"Loaded tracker_T_3s with shape: {tracker_T_3s.shape}")
    except Exception as e:
        print(f"Error loading tracker poses: {e}")
        exit(1)
    
    # Make sure tracker data has enough entries for our valid groups
    max_tracker_idx = max(tracker_indices.values())
    if max_tracker_idx >= tracker_T_3s.shape[0]:
        print(f"Warning: Some tracker indices ({max_tracker_idx}) exceed available tracker data ({tracker_T_3s.shape[0]})")
        # Adjust valid_groups if needed
        valid_groups = [g for g in valid_groups if tracker_indices[g] < tracker_T_3s.shape[0]]
        if not valid_groups:
            print("No valid groups remain after filtering for tracker data availability")
            exit(1)
        print(f"Reduced to valid groups: {valid_groups}")
    
    print(f"Number of valid robot states: {len(valid_robot_states)}")
    
    if Setup603:
        end_T_cam = np.array([[0.99999378, 0.00218076, -0.00277091, 0.04262605],
                            [-0.00216497, 0.99998148, 0.00568824, 0.0067556],
                            [0.00278326, -0.00568221, 0.99997998, 0.12970971],
                            [0, 0, 0, 1]])
    else:
        if SetupMechEye:
            # 2025-02-05 mecheye
            end_T_cam = np.array([[ 6.52032141e-03,  9.99931118e-01,  9.75935357e-03,  7.34302525e-02],
                                  [-9.99978640e-01,  6.52442644e-03, -3.88847118e-04, -4.19321620e-02],
                                  [-4.52494518e-04, -9.75660970e-03,  9.99952301e-01,  7.64250999e-02],
                                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        else:
            if OnlyGroup3n4:
                # 16w 02-24
                end_T_cam = np.array([[-0.00280774, -0.99993985,  0.01060246,  0.03759239],
                                      [ 0.96403859, -0.00552423, -0.26570487,  0.11280878],
                                      [ 0.26574746,  0.00947515,  0.96399612,  0.12180479],
                                      [ 0.,          0.,          0.,          1.        ]])
            else:
                # 16w 01-17
                # end_T_cam = np.array([[-4.69196201e-04,  9.99961201e-01, -8.79633495e-03, -3.60227528e-02],
                #                       [-9.65813707e-01,  1.82719465e-03,  2.59230678e-01, -1.12813121e-01],
                #                       [ 2.59236693e-01,  8.61725091e-03,  9.65775378e-01,  1.20060964e-01],
                #                       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
                # 16w 02-04
                end_T_cam = np.array([[-3.30782133e-04,  9.99963029e-01, -8.59247147e-03, -3.61813900e-02],
                                      [-9.65561487e-01,  1.91616571e-03,  2.60167912e-01, -1.12514682e-01],
                                      [ 2.60174758e-01,  8.38261843e-03,  9.65525156e-01,  1.20075923e-01],
                                      [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
                # 16w 02-24
                # end_T_cam = np.array([[-0.15574418, -0.98077204,  0.11760079, -0.14335974],
                #                       [ 0.90361027, -0.18955117, -0.38413388,  0.52448638],
                #                       [ 0.39903914,  0.04643866,  0.91575718, -0.40286519],
                #                       [ 0.,          0.,          0.,          1.        ]])

    # Process tracker data for valid groups using valid_group indices
    tracker_T_camidxs = []
    
    # We need to iterate by index through valid_groups, not their values
    for i, group in enumerate(valid_groups):
        robot_state = valid_robot_states[i]
        tracker_idx = tracker_indices[group]
        
        # Make sure robot state and tracker index are valid
        if tracker_idx < tracker_T_3s.shape[0]:
            f = robot_new.forward(np.array(robot_state))
            Ts = robot_new.ts
            Link3TEnd_i = np.eye(4)
            for j in range(3, 6):
                Link3TEnd_i = Link3TEnd_i.dot(Ts[j].t_4_4)
            current_tracker_T_3 = tracker_T_3s[tracker_idx]
            tracker_T_camidx = current_tracker_T_3.dot(Link3TEnd_i).dot(end_T_cam)
            tracker_T_camidxs.append(tracker_T_camidx)
        else:
            print(f"Warning: Tracker index {tracker_idx} for group {group} exceeds available data")

    if not tracker_T_camidxs:
        print("No valid tracker transformations computed. Cannot proceed.")
        exit(1)

    tracker_T_cam1 = tracker_T_camidxs[0]
    K3DoF_cam1_T_camidxs = []
    
    for idx in range(len(tracker_T_camidxs) - 1):
        tracker_T_camidx = tracker_T_camidxs[idx + 1]
        K3DoF_cam1_T_camidx = np.linalg.inv(tracker_T_cam1).dot(tracker_T_camidx)

        if idx + 1 < len(valid_ply_files):  # Make sure we have a valid ply file
            ply_file = valid_ply_files[idx + 1]
            pointcloud_2 = o3d.io.read_point_cloud(ply_file)
            K = K3DoF_cam1_T_camidx.copy()
            if Setup603 or SetupMechEye:
                K[:3, 3] = 1000.0 * K[:3, 3]
            else:  # No scaling needed when point cloud is in meters
                K[:3, 3] = K[:3, 3]
            pointcloud_2_calib = copy.deepcopy(pointcloud_2).transform(K)
            new_ply_file = ply_file.replace(".ply", "_trans_3DoF.ply")
            o3d.io.write_point_cloud(new_ply_file, pointcloud_2_calib)

        K3DoF_cam1_T_camidxs.append(K3DoF_cam1_T_camidx)

    # Now we compute the errors using our aligned data
    errors_R_K6DoF = []
    errors_t_K6DoF = []
    errors_R_K3DoF = []
    errors_t_K3DoF = []
    
    # Make sure we only process as many transformations as we have valid data for
    num_valid_transforms = min(len(cam_cam1_T_camidxs), len(K6DoF_cam1_T_camidxs), len(K3DoF_cam1_T_camidxs))
    
    for idx in range(num_valid_transforms):
        cam_cam1_T_cami = cam_cam1_T_camidxs[idx]
        K6DoF_cam1_T_cami = K6DoF_cam1_T_camidxs[idx]
        K3DoF_cam1_T_cami = K3DoF_cam1_T_camidxs[idx]
        
        # Format the appearance to make the code easy to read
        error_R_K6DoF, error_t_K6DoF = compute_transformation_diff(
            K6DoF_cam1_T_cami[:3, :3], K6DoF_cam1_T_cami[:3, 3:], 
            cam_cam1_T_cami[:3, :3], cam_cam1_T_cami[:3, 3:]
        )
        error_R_K3DoF, error_t_K3DoF = compute_transformation_diff(
            K3DoF_cam1_T_cami[:3, :3], K3DoF_cam1_T_cami[:3, 3:], 
            cam_cam1_T_cami[:3, :3], cam_cam1_T_cami[:3, 3:]
        )

        errors_R_K6DoF.append(error_R_K6DoF)
        errors_t_K6DoF.append(error_t_K6DoF * 1000)
        errors_R_K3DoF.append(error_R_K3DoF)
        errors_t_K3DoF.append(error_t_K3DoF * 1000)

    # Only create plots if we have valid data
    if errors_R_K6DoF and errors_R_K3DoF:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # First subplot - Rotation Error
        ax1.plot(range(len(errors_R_K6DoF)), errors_R_K6DoF, label="6DOF_R_error")
        ax1.plot(range(len(errors_R_K3DoF)), errors_R_K3DoF, label="Ours_R_error")
        ax1.set_title("Rotation Error")
        ax1.set_ylim(0, max(max(errors_R_K6DoF), max(errors_R_K3DoF)) * 1.1)  # Dynamic Y limit
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Error (deg.)')
        ax1.legend()

        # Second subplot - Translation Error
        ax2.plot(range(len(errors_t_K6DoF)), errors_t_K6DoF, label="6DOF_t_error")
        ax2.plot(range(len(errors_t_K3DoF)), errors_t_K3DoF, label="Ours_t_error")
        ax2.set_title("Translation Error")
        ax2.set_ylim(0, max(max(errors_t_K6DoF), max(errors_t_K3DoF)) * 1.1)  # Dynamic Y limit
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Error (mm)')
        ax2.legend()

        plt.tight_layout()  # Adjusts subplot params for better spacing
        plt.subplots_adjust(hspace=0.4)  # Add space between subplots

        # Save the figure to a file since the backend doesn't support display
        plt.savefig('error_comparison_plot.png', dpi=300)

        # You can optionally still try to show it
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display plot: {e}")
            print("Plot has been saved to 'error_comparison_plot.png'")

    print("errors_R_K6DoF", errors_R_K6DoF)
    print("errors_t_K6DoF", errors_t_K6DoF)
    print("errors_R_K3DoF", errors_R_K3DoF)
    print("errors_t_K3DoF", errors_t_K3DoF)