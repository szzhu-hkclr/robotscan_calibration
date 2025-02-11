#include "RobotSerial.hpp"
#include <cmath>

RobotSerial::RobotSerial(const std::vector<DHParams>& dh_params) : dh_params_(dh_params) {}

cv::Mat RobotSerial::compute_dh_transform(const DHParams& dh, double joint_angle) const {
    // Compute the transformation using standard DH parameters.
    // The effective theta is the sum of the fixed offset and the provided joint angle.
    double theta = dh.theta + joint_angle;
   
    double cos_theta = cos(theta);
    double sin_theta = sin(theta);
    double cos_alpha = cos(dh.alpha);
    double sin_alpha = sin(dh.alpha);

    return (cv::Mat_<double>(4, 4) <<
        cos_theta, -sin_theta * cos_alpha,  sin_theta * sin_alpha, dh.a * cos_theta,
        sin_theta,  cos_theta * cos_alpha, -cos_theta * sin_alpha, dh.a * sin_theta,
        0.0,        sin_alpha,              cos_alpha,             dh.d,
        0.0,        0.0,                    0.0,                   1.0);
}

cv::Mat RobotSerial::forward(const std::vector<double>& joint_angles) {
    // Check that joint_angles has the same number of elements as DH parameters.
    if (joint_angles.size() != dh_params_.size()) {
        std::cout << "joint_angles size: " << joint_angles.size() << std::endl;
        std::cout << "dh_params size: " << dh_params_.size() << std::endl;
        throw std::runtime_error("Mismatch between number of joint angles and DH parameter rows.");
    }
    
    // Compute the individual transformation matrices (non-cumulative) for each joint.
    std::vector<cv::Mat> Ts_individual;
    Ts_individual.reserve(dh_params_.size());
    for (size_t i = 0; i < dh_params_.size(); ++i) {
        cv::Mat Ti = compute_dh_transform(dh_params_[i], joint_angles[i]);
        Ts_individual.push_back(Ti);
    }
    
    // Cache the individual transforms 
    ts_ = Ts_individual;
    
    // Compute the cumulative transformation (i.e., the end-effector frame)
    cv::Mat T_cumulative = cv::Mat::eye(4, 4, CV_64F);
    for (size_t i = 0; i < Ts_individual.size(); ++i) {
        T_cumulative = T_cumulative * Ts_individual[i];
    }
    
    return T_cumulative;
}

std::vector<cv::Mat> RobotSerial::get_ts() const {
    // Returns the cached individual transformation matrices (non-cumulative) computed in forward().
    return ts_;
}