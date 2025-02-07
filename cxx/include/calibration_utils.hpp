#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct CalibrationResult {
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    double reprojection_error;
};

CalibrationResult calibrate_camera(
    const std::vector<std::vector<cv::Point3f>>& object_points,
    const std::vector<std::vector<cv::Point2f>>& image_points,
    cv::Size image_size);

std::vector<std::vector<double>> read_joint_angles(const std::string& file_path);