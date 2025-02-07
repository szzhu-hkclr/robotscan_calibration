#include "RobotSerial.hpp"

RobotSerial::RobotSerial(const std::vector<std::vector<double>>& dh_params) 
    : dh_params_(dh_params) {}

cv::Mat RobotSerial::forward(const std::vector<double>& joint_angles) {
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    for (size_t i = 0; i < dh_params_.size(); ++i) {
        const auto& dh = dh_params_[i];
        double d = dh[0];
        double a = dh[1];
        double alpha = dh[2];
        double theta = dh[3] + joint_angles[i];

        double cos_theta = cos(theta);
        double sin_theta = sin(theta);
        double cos_alpha = cos(alpha);
        double sin_alpha = sin(alpha);

        cv::Mat Ti = (cv::Mat_<double>(4, 4) <<
            cos_theta, -sin_theta*cos_alpha,  sin_theta*sin_alpha, a*cos_theta,
            sin_theta,  cos_theta*cos_alpha, -cos_theta*sin_alpha, a*sin_theta,
            0.0,        sin_alpha,            cos_alpha,           d,
            0.0,        0.0,                  0.0,                 1.0);

        T = T * Ti;
    }
    return T;
}