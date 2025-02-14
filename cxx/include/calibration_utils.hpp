#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "cnpy.h"

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

void compute_camera_extrinsics(const std::vector<std::vector<cv::Point3f>>& object_points,
                               const std::vector<std::vector<cv::Point2f>>& image_points,
                               const cv::Mat& camera_matrix,
                               const cv::Mat& dist_coeffs,
                               std::vector<cv::Mat>& R_target2cam,
                               std::vector<cv::Mat>& T_target2cam);

std::vector<cv::Mat> load_tracker_poses(const std::string &tracker_pose_file);

std::vector<cv::Mat> compute_camera_poses(
    const std::vector<std::vector<cv::Point2f>> &chessboard_corners,
    cv::Size pattern_size,
    float square_size,
    const cv::Mat &intrinsic_matrix,
    const cv::Mat &dist,
    bool Testing = false);                      