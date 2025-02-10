#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdexcept>

struct DHParams {
    double d;
    double a;
    double alpha;
    double theta;
};

class RobotSerial {
public:
    RobotSerial(const std::vector<DHParams>& dh_params);
    cv::Mat forward(const std::vector<double>& joint_angles);
    
private:
    cv::Mat compute_dh_transform(const DHParams& dh, double joint_angle) const;
    std::vector<DHParams> dh_params_;
};