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