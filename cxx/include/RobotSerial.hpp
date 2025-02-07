#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdexcept>

class RobotSerial {
public:
    RobotSerial(const std::vector<std::vector<double>>& dh_params);
    cv::Mat forward(const std::vector<double>& joint_angles);
    
private:
    std::vector<std::vector<double>> dh_params_;
};