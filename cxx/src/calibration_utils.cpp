#include "calibration_utils.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

CalibrationResult calibrate_camera(
    const std::vector<std::vector<cv::Point3f>>& object_points,
    const std::vector<std::vector<cv::Point2f>>& image_points,
    cv::Size image_size) {
    
    CalibrationResult result;
    std::vector<cv::Mat> rvecs, tvecs;
    
    // Use 0 for flags so that calibration settings are similar to Python defaults.
    result.reprojection_error = cv::calibrateCamera(
        object_points, image_points, image_size,
        result.camera_matrix, result.dist_coeffs,
        rvecs, tvecs, 0
    );
    
    result.rvecs = rvecs;
    result.tvecs = tvecs;
    return result;
}

std::vector<std::vector<double>> read_joint_angles(const std::string& file_path) {
    std::vector<std::vector<double>> angles_list;
    std::ifstream fin(file_path);
    if (!fin.is_open()) {
        throw std::runtime_error("Unable to open file: " + file_path);
    }
    
    std::string line;
    while (std::getline(fin, line)) {
        std::vector<double> angles;
        std::istringstream ss(line);
        std::string token;
        
        // Split the line on commas
        while (std::getline(ss, token, ',')) {
            std::istringstream token_stream(token);
            double angle;
            token_stream >> angle; // convert each token to double
            angles.push_back(angle);
        }
        
        if (!angles.empty()) {
            angles_list.push_back(angles);
        }
    }
    return angles_list;
}

void compute_camera_extrinsics(const std::vector<std::vector<cv::Point3f>>& object_points,
                               const std::vector<std::vector<cv::Point2f>>& image_points,
                               const cv::Mat& camera_matrix,
                               const cv::Mat& dist_coeffs,
                               std::vector<cv::Mat>& R_target2cam,
                               std::vector<cv::Mat>& T_target2cam) {
    for (size_t i = 0; i < image_points.size(); ++i) {
        cv::Mat rvec, tvec;
        cv::solvePnP(object_points[i], image_points[i], camera_matrix, dist_coeffs, rvec, tvec);
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        R_target2cam.push_back(R);
        T_target2cam.push_back(tvec);
    }
}

std::vector<cv::Mat> load_tracker_poses(const std::string& tracker_pose_file) {
    std::vector<cv::Mat> tracker_poses;
    cnpy::NpyArray arr = cnpy::npy_load(tracker_pose_file);

    // Check that the NpyArray has dimensions (N, 4, 4)
    if (arr.shape.size() != 3 || arr.shape[1] != 4 || arr.shape[2] != 4) {
        std::cerr << "Error: Expected a 3D array with dimensions (N, 4, 4)" << std::endl;
        throw std::runtime_error("Invalid npy file shape");
    }

    size_t num_matrices = arr.shape[0];
    size_t elements_per_matrix = 16;  // 4x4

    // Determine if data is stored as double or float using the word size
    if (arr.word_size == sizeof(double)) {
        double* data = arr.data<double>();
        for (size_t i = 0; i < num_matrices; ++i) {
            cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    pose.at<double>(r, c) = data[i * elements_per_matrix + r * 4 + c];
                }
            }
            tracker_poses.push_back(pose);
        }
    } else if (arr.word_size == sizeof(float)) {
        float* data = arr.data<float>();
        for (size_t i = 0; i < num_matrices; ++i) {
            cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    // Cast float to double when populating the cv::Mat for consistency
                    pose.at<double>(r, c) = static_cast<double>(data[i * elements_per_matrix + r * 4 + c]);
                }
            }
            tracker_poses.push_back(pose);
        }
    } else {
        std::cerr << "Error: Unrecognized data type in npy file." << std::endl;
        throw std::runtime_error("Unsupported npy data type");
    }

    return tracker_poses;
}

std::vector<cv::Mat> compute_camera_poses(
    const std::vector<std::vector<cv::Point2f>>& chessboard_corners,
    cv::Size pattern_size,
    float square_size,
    const cv::Mat& intrinsic_matrix,
    const cv::Mat& dist,
    bool Testing)
{
    std::vector<cv::Mat> cam_T_chesss; // This will hold the 4x4 transformation matrices.
    
    // Build the object points (points in real space) for the chessboard
    std::vector<cv::Point3f> object_points;
    for (int i = 0; i < pattern_size.height; ++i) {
        for (int j = 0; j < pattern_size.width; ++j) {
            object_points.push_back(cv::Point3f(j * square_size, i * square_size, 0.0f));
        }
    }
    
    int iteration = 1;
    
    // Loop over each set of chessboard corners.
    for (const auto& corners : chessboard_corners)
    {
        cv::Mat rvec, tvec;
        bool success = cv::solvePnP(object_points, corners, intrinsic_matrix, dist, rvec, tvec);
        if (!success) {
            std::cerr << "solvePnP failed for iteration " << iteration << std::endl;
            iteration++;
            continue;
        }
    
        if (Testing) {
            std::cout << "Current iteration: " << iteration 
                      << " out of " << chessboard_corners.size() << " iterations." << std::endl;
            std::cout << "rvec: " << rvec.t() << std::endl;
            std::cout << "rvec[0]: " << rvec.at<double>(0) << std::endl;
            std::cout << "rvec[1]: " << rvec.at<double>(1) << std::endl;
            std::cout << "rvec[2]: " << rvec.at<double>(2) << std::endl;
            std::cout << "--------------------" << std::endl;
        }
    
        // Convert the rotation vector to a rotation matrix.
        cv::Mat R;
        cv::Rodrigues(rvec, R);
    
        // Build the 4x4 transformation matrix.
        cv::Mat cam_T_chess = cv::Mat::eye(4, 4, R.type());
        // Insert rotation matrix into the top-left 3x3 block.
        R.copyTo(cam_T_chess(cv::Rect(0, 0, 3, 3)));
        // Insert translation vector into the top-right 3x1 block.
        cam_T_chess.at<double>(0, 3) = tvec.at<double>(0);
        cam_T_chess.at<double>(1, 3) = tvec.at<double>(1);
        cam_T_chess.at<double>(2, 3) = tvec.at<double>(2);
    
        cam_T_chesss.push_back(cam_T_chess);
        iteration++;
    }
    
    return cam_T_chesss;
}
