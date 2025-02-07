#include "calibration_utils.hpp"

CalibrationResult calibrate_camera(
    const std::vector<std::vector<cv::Point3f>>& object_points,
    const std::vector<std::vector<cv::Point2f>>& image_points,
    cv::Size image_size) {
    
    CalibrationResult result;
    std::vector<cv::Mat> rvecs, tvecs;
    
    result.reprojection_error = cv::calibrateCamera(
        object_points, image_points, image_size,
        result.camera_matrix, result.dist_coeffs,
        rvecs, tvecs, cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5
    );
    
    result.rvecs = rvecs;
    result.tvecs = tvecs;
    return result;
}

std::vector<std::vector<double>> read_joint_angles(const std::string& file_path) {
    std::vector<std::vector<double>> angles_list;
    std::ifstream fin(file_path);
    std::string line;
    
    while (std::getline(fin, line)) {
        std::istringstream iss(line);
        std::vector<double> angles;
        double angle;
        
        while (iss >> angle) {
            angles.push_back(angle);
        }
        
        if (!angles.empty()) {
            angles_list.push_back(angles);
        }
    }
    return angles_list;
}