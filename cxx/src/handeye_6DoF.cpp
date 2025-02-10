#include "handeye_6DoF.hpp"
#include "RobotSerial.hpp"
#include "calibration_utils.hpp"
#include "image_utils.hpp"
#include <iostream>
#include <stdexcept>

// Function to perform the 6DOF hand–eye calibration.
int run6DoFHandEyeCalibration(const Config &config) {
    // Convert pattern size into a cv::Size from the config.
    cv::Size pattern_size(config.pattern_size[0], config.pattern_size[1]);
    RobotSerial robot(config.dh_params);

    std::vector<cv::Mat> REnd2Base, TEnd2Base;
    std::vector<std::vector<cv::Point2f>> all_image_points;
    std::vector<std::vector<cv::Point3f>> all_object_points;
    cv::Size image_size; 

    // Process each group from the configuration.
    for (const auto &group : config.group_lists) {
        std::string image_group_folder = config.image_folder + "/" + group;
        auto images = read_images(image_group_folder);
        if (images.empty()) {
            std::cerr << "No images found in folder: " << image_group_folder << std::endl;
            continue;
        }
        
        // Initialize image size if not already set.
        if (image_size.width == 0 && image_size.height == 0) {
            image_size = images[0].size();
        }
        
        auto cb_result = find_chessboard_corners(images, pattern_size, config.square_size, config.show_corners);
        if (cb_result.image_points.empty()) continue;

        // Read the joint angles file for this group.
        std::string joint_file = config.pose_folder + "/" + group + ".txt";
        auto joints_records = read_joint_angles(joint_file);
        if (joints_records.empty()) {
            std::cerr << "No joint angles found in file: " << joint_file << std::endl;
            continue;
        }

        // Use each valid image index to select the corresponding joint angles.
        // (It is assumed that the order of valid corners and joint records is the same.)
        for (size_t k = 0; k < cb_result.valid_indices.size(); ++k) {
            int idx = cb_result.valid_indices[k];
            if (idx >= static_cast<int>(joints_records.size())) {
                std::cerr << "Not enough joint angle records for index " << idx << std::endl;
                continue;
            }

            // Compute EE forward kinematics.
            cv::Mat T = robot.forward(joints_records[idx]).back();
            cv::Mat R = T(cv::Rect(0, 0, 3, 3)).clone();
            cv::Mat t = T(cv::Rect(3, 0, 1, 3)).clone();

            REnd2Base.push_back(R);
            TEnd2Base.push_back(t);
        }

        // Append chessboard detection information.
        all_image_points.insert(all_image_points.end(), 
                                cb_result.image_points.begin(),
                                cb_result.image_points.end());
        all_object_points.insert(all_object_points.end(),
                                 cb_result.object_points.begin(),
                                 cb_result.object_points.end());
    }

    // Calibrate the camera intrinsics.
    CalibrationResult calib_result = calibrate_camera(all_object_points, all_image_points, image_size);
    if (config.show_projection_error) {
        std::cout << "Reprojection error: " << calib_result.reprojection_error << std::endl;
    }

    // Convert rotation vectors to rotation matrices.
    std::vector<cv::Mat> RTarget2Cam, TTarget2Cam;
    for (size_t i = 0; i < calib_result.rvecs.size(); ++i) {
        cv::Mat R;
        cv::Rodrigues(calib_result.rvecs[i], R);
        RTarget2Cam.push_back(R);
        TTarget2Cam.push_back(calib_result.tvecs[i]);
    }

    // Perform hand–eye calibration using cv::CALIB_HAND_EYE_TSAI method.
    cv::Mat R_cam2gripper, t_cam2gripper;
    cv::calibrateHandEye(REnd2Base, TEnd2Base, RTarget2Cam, TTarget2Cam, 
                         R_cam2gripper, t_cam2gripper, cv::CALIB_HAND_EYE_TSAI);

    // Compose the final 4x4 transformation matrix.
    cv::Mat end_T_cam = cv::Mat::eye(4, 4, CV_64F);
    R_cam2gripper.copyTo(end_T_cam(cv::Rect(0, 0, 3, 3)));
    t_cam2gripper.copyTo(end_T_cam(cv::Rect(3, 0, 1, 3)));

    std::cout << "Hand-eye calibration result (end_T_cam):\n" << end_T_cam << std::endl;
    std::cout << "Camera Intrinsics:\n" << calib_result.camera_matrix << std::endl;
    std::cout << "Distortion Coefficients:\n" << calib_result.dist_coeffs << std::endl;

    return 0;
}