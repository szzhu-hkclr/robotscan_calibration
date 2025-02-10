#include "RobotSerial.hpp"
#include "json_utils.hpp"
#include "image_utils.hpp"
#include "calibration_utils.hpp"

int runNewKinHandEyeCalibration(const Config &config) {

    // Load images from each group using read_images() provided in image_utils.cpp.
    std::vector<cv::Mat> images;
    for (const std::string &group : config.group_lists) {
        std::string group_folder = config.image_folder + "/" + group;
        std::vector<cv::Mat> group_images = read_images(group_folder);
        images.insert(images.end(), group_images.begin(), group_images.end());
    }
    
    // Detect chessboard corners using function from image_utils.hpp.
    cv::Size pattern_size(config.pattern_size[0], config.pattern_size[1]);
    ChessboardResult chessboard_result = find_chessboard_corners(images, pattern_size,
                                                                   config.square_size,
                                                                   config.show_corners);
    if (chessboard_result.image_points.empty()) {
        std::cerr << "No valid chessboard corners found!" << std::endl;
        return -1;
    }
    
    // Calibrate camera using calibration_utils.cpp function.
    CalibrationResult calib_result = calibrate_camera(chessboard_result.object_points,
                                                       chessboard_result.image_points,
                                                       images[0].size());
    if (config.show_projection_error)                                                    
        std::cout << "Reprojection error: " << calib_result.reprojection_error << std::endl;
    
    // Compute camera extrinsics (target-to-camera) using the helper function moved to calibration_utils.cpp.
    std::vector<cv::Mat> R_target2cam, T_target2cam;
    compute_camera_extrinsics(chessboard_result.object_points, chessboard_result.image_points,
                              calib_result.camera_matrix, calib_result.dist_coeffs,
                              R_target2cam, T_target2cam);
    
    // Load tracker poses using the helper function in calibration_utils.cpp.
    std::vector<cv::Mat> tracker_T_3s = load_tracker_poses(config.tracker_pose_file);
    
    // Compute robot poses using the RobotSerial class.
    RobotSerial robot(config.dh_params);
    std::vector<cv::Mat> R_end2base, T_end2base;
    
    // For each group, load joint angles file and compute forward kinematics.
    for (size_t g = 0; g < config.group_lists.size(); ++g) {
        std::string joints_file = config.pose_folder + "/" + config.group_lists[g] + ".txt";
        auto joints_records = read_joint_angles(joints_file);
        if (joints_records.empty()) {
            std::cerr << "No joint angles found in file: " << joints_file << std::endl;
            continue;
        }
        for (const auto &joint_angles : joints_records) {
            auto Ts = robot.forward(joint_angles);
            cv::Mat T_3_end = cv::Mat::eye(4, 4, CV_64F);
            for (int j = 3; j < 6; ++j)
                T_3_end = T_3_end * Ts[j];
            cv::Mat T = tracker_T_3s[g] * T_3_end;
            R_end2base.push_back(T(cv::Rect(0, 0, 3, 3)).clone());
            T_end2base.push_back(T(cv::Rect(3, 0, 1, 3)).clone());
        }
    }
    
    // Perform hand–eye calibration using method 4.
    cv::Mat R_cam2gripper, T_cam2gripper;
    cv::calibrateHandEye(R_end2base, T_end2base, R_target2cam, T_target2cam,
                         R_cam2gripper, T_cam2gripper, cv::CALIB_HAND_EYE_TSAI);
    
    // Construct the 4x4 transformation matrix for the hand-eye calibration result.
    cv::Mat end_T_cam = cv::Mat::eye(4, 4, CV_64F);
    R_cam2gripper.copyTo(end_T_cam(cv::Rect(0, 0, 3, 3)));
    T_cam2gripper.copyTo(end_T_cam(cv::Rect(3, 0, 1, 3)));
    
    std::cout << "Hand-eye calibration result (end_T_cam):\n" << end_T_cam << std::endl;
    return 0;
}