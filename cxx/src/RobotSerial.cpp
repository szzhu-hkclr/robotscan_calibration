#include "RobotSerial.hpp"
#include <cmath>

RobotSerial::RobotSerial(const std::vector<DHParams>& dh_params) : dh_params_(dh_params) {}

cv::Mat RobotSerial::compute_dh_transform(const DHParams& dh, double joint_angle) const {
    // Assuming the provided joint angle is in radians.
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

std::vector<cv::Mat> RobotSerial::forward(const std::vector<double>& joint_angles) {
    // Check that joint_angles has the same number of elements as dh parameters.
    if (joint_angles.size() != dh_params_.size()) {
        std::cout << "joint_angles size:\n" << joint_angles.size() << std::endl;
        std::cout << "dh_params size :\n" << dh_params_.size() << std::endl;
        throw std::runtime_error("Mismatch between number of joint angles and DH parameter rows.");
    }
    
    std::vector<cv::Mat> Ts;
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    for (size_t i = 0; i < dh_params_.size(); ++i) {
        cv::Mat Ti = compute_dh_transform(dh_params_[i], joint_angles[i]);
        T = T * Ti;
        Ts.push_back(T.clone());
    }

    return Ts;
}