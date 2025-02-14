#include "RobotSerial.hpp"
#include "json_utils.hpp"
#include "image_utils.hpp"
#include "calibration_utils.hpp"

// Helper function: convert a 2D std::vector<double> (3x3) to cv::Mat.
cv::Mat convertToMat3x3(const std::vector<std::vector<double>>& vec) {
    cv::Mat mat(3, 3, CV_64F);
    for (size_t i = 0; i < vec.size(); ++i) {
        for (size_t j = 0; j < vec[i].size(); ++j) {
            mat.at<double>(i, j) = vec[i][j];
        }
    }
    return mat;
}

// Helper function: convert a 2D std::vector<double> (4x4) to cv::Mat.
cv::Mat convertToMat4x4(const std::vector<std::vector<double>>& vec) {
    cv::Mat mat(4, 4, CV_64F);
    for (size_t i = 0; i < vec.size(); ++i) {
        for (size_t j = 0; j < vec[i].size(); ++j) {
            mat.at<double>(i, j) = vec[i][j];
        }
    }
    return mat;
}

std::vector<double> rotationMatrixToEulerAngles(const cv::Mat &R) {
    double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) +
                          R.at<double>(1, 0) * R.at<double>(1, 0));
    bool singular = sy < 1e-6;
    double x, y, z;
    if (!singular) {
        x = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    } else {
        x = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }
    return { x * 180.0 / CV_PI, y * 180.0 / CV_PI, z * 180.0 / CV_PI };
}

// Computes average absolute difference in Euler angles between two rotations.
double getAngularError(const cv::Mat &R_gt, const cv::Mat &R_est) {
    std::vector<double> eu_gt = rotationMatrixToEulerAngles(R_gt);
    std::vector<double> eu_est = rotationMatrixToEulerAngles(R_est);
    double sum = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        sum += std::abs(eu_gt[i] - eu_est[i]);
    }
    return sum / 3.0;
}

// Computes transformation difference error: rotation error and translation error.
std::pair<double, double> compute_transformation_diff(const cv::Mat &R_est, const cv::Mat &t_est,
                                                        const cv::Mat &R_gt, const cv::Mat &t_gt) {
    double rot_error = getAngularError(R_gt, R_est);
    cv::Mat diff = t_gt - t_est;
    double trans_error = cv::norm(diff);
    return std::make_pair(rot_error, trans_error);
}

int runevaluateHandEyeError(const Config &config) {
    try {
        // Read images from the file path provided by config.image_files.
        std::vector<cv::Mat> images = read_images(config.image_files);
        if (images.empty()) {
            std::cerr << "No images found in: " << config.image_files << std::endl;
            return -1;
        }

        // Detect chessboard corners in the images.
        cv::Size pattern_size(config.pattern_size[0], config.pattern_size[1]);
        ChessboardResult chessboard_result = find_chessboard_corners(images, pattern_size, static_cast<float>(config.square_size), config.show_corners);
        if (chessboard_result.image_points.empty()) {
            std::cerr << "No chessboard corners detected." << std::endl;
            return -1;
        }

        // Convert intrinsic data to cv::Mat.
        cv::Mat intrinsic_matrix = convertToMat3x3(config.intrinsic_matrix);

        // For this example, assume zero distortion.
        cv::Mat dist = cv::Mat::zeros(5, 1, CV_64F);

        // Compute camera poses with respect to the chessboard.
        std::vector<cv::Mat> cam_T_chesss = compute_camera_poses(chessboard_result.image_points, pattern_size, static_cast<float>(config.square_size), intrinsic_matrix, dist, false);
        if (cam_T_chesss.empty()) {
            std::cerr << "No valid camera poses computed." << std::endl;
            return -1;
        }

        // Use the first computed camera pose as reference.
        cv::Mat cam1_T_chess = cam_T_chesss[0];

        // Compute relative camera transforms (from the first pose).
        std::vector<cv::Mat> cam_relative_transforms;
        for (size_t i = 0; i < cam_T_chesss.size(); ++i) {
            cv::Mat T_rel = cam1_T_chess.inv() * cam_T_chesss[i];
            cam_relative_transforms.push_back(T_rel);
        }

        // Read robot joint states from the file provided by config.robotfile_path.
        std::vector<std::vector<double>> robot_states = read_joint_angles(config.robotfile_path);
        if (robot_states.empty()) {
            std::cerr << "No robot states loaded from: " << config.robotfile_path << std::endl;
            return -1;
        }

        // Create the end-effector to camera transformation matrix.
        cv::Mat end_T_cam_ori = convertToMat4x4(config.end_T_cam_ori);

        // Build RobotSerial objects for 6DoF and new kinematics using the parsed DH parameters.
        RobotSerial robot6DoF(config.dh_params);
        RobotSerial robotNewkin(config.dh_params_new);

        // Compute forward kinematics for each robot state for 6DoF.
        std::vector<cv::Mat> base_T_camidxs_6DoF;
        for (size_t i = 0; i < robot_states.size(); ++i) {
            cv::Mat T_robot = robot6DoF.forward(robot_states[i]);
            cv::Mat T_cam = T_robot * end_T_cam_ori;
            base_T_camidxs_6DoF.push_back(T_cam);
        }
        cv::Mat base_T_cam1_6DoF = base_T_camidxs_6DoF[0];
        std::vector<cv::Mat> relative_transforms_6DoF;
        for (size_t i = 0; i < base_T_camidxs_6DoF.size(); ++i) {
            cv::Mat T_rel = base_T_cam1_6DoF.inv() * base_T_camidxs_6DoF[i];
            relative_transforms_6DoF.push_back(T_rel);
        }

        // Compute forward kinematics using new kinematics.
        std::vector<cv::Mat> base_T_camidxs_newkin;
        for (size_t i = 0; i < robot_states.size(); ++i) {
            cv::Mat T_robot = robotNewkin.forward(robot_states[i]);
            cv::Mat T_cam = T_robot * end_T_cam_ori;
            base_T_camidxs_newkin.push_back(T_cam);
        }
        cv::Mat base_T_cam1_newkin = base_T_camidxs_newkin[0];
        std::vector<cv::Mat> relative_transforms_newkin;
        for (size_t i = 0; i < base_T_camidxs_newkin.size(); ++i) {
            cv::Mat T_rel = base_T_cam1_newkin.inv() * base_T_camidxs_newkin[i];
            relative_transforms_newkin.push_back(T_rel);
        }

        // Calculate errors between the camera computed relative transforms and those computed by robot kinematics.
        std::vector<double> errors_R_6DoF, errors_t_6DoF;
        std::vector<double> errors_R_newkin, errors_t_newkin;
        size_t n = std::min(cam_relative_transforms.size(), relative_transforms_6DoF.size());
        for (size_t i = 0; i < n; ++i) {
            cv::Mat R_cam = cam_relative_transforms[i](cv::Rect(0, 0, 3, 3));
            cv::Mat t_cam = cam_relative_transforms[i](cv::Rect(3, 0, 1, 3));
            cv::Mat R_6D = relative_transforms_6DoF[i](cv::Rect(0, 0, 3, 3));
            cv::Mat t_6D = relative_transforms_6DoF[i](cv::Rect(3, 0, 1, 3));
            cv::Mat R_new = relative_transforms_newkin[i](cv::Rect(0, 0, 3, 3));
            cv::Mat t_new = relative_transforms_newkin[i](cv::Rect(3, 0, 1, 3));
            
            std::pair<double, double> diff6D = compute_transformation_diff(R_6D, t_6D, R_cam, t_cam);
            std::pair<double, double> diffNew = compute_transformation_diff(R_new, t_new, R_cam, t_cam);
            errors_R_6DoF.push_back(diff6D.first);
            errors_t_6DoF.push_back(diff6D.second * 1000.0); // convert to mm
            errors_R_newkin.push_back(diffNew.first);
            errors_t_newkin.push_back(diffNew.second * 1000.0);
        }

        // Print error results.
        std::cout << "Rotation Errors (6DoF): ";
        for (size_t i = 0; i < errors_R_6DoF.size(); ++i) {
            std::cout << errors_R_6DoF[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Translation Errors (6DoF, mm): ";
        for (size_t i = 0; i < errors_t_6DoF.size(); ++i) {
            std::cout << errors_t_6DoF[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Rotation Errors (NewKinematics): ";
        for (size_t i = 0; i < errors_R_newkin.size(); ++i) {
            std::cout << errors_R_newkin[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Translation Errors (NewKinematics, mm): ";
        for (size_t i = 0; i < errors_t_newkin.size(); ++i) {
            std::cout << errors_t_newkin[i] << " ";
        }
        std::cout << std::endl;
        
        // Save the error results in a JSON file.
        nlohmann::json error_results;
        error_results["errors_R_6DoF"] = errors_R_6DoF;
        error_results["errors_t_6DoF"] = errors_t_6DoF;
        error_results["errors_R_NewKinematics"] = errors_R_newkin;
        error_results["errors_t_NewKinematics"] = errors_t_newkin;
        
        std::ofstream ofs("error_results.json");
        ofs << error_results.dump(4);
        ofs.close();
    } catch (std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}