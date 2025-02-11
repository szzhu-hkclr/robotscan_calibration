// RobotSerial.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdexcept>
#include <iostream>

struct DHParams {
    double d;
    double a;
    double alpha;
    double theta;
};

class RobotSerial {
public:
    // Constructor: accepts a vector of DHParams.
    RobotSerial(const std::vector<DHParams>& dh_params);
    
    // forward: sets the joint angles, computes transformation matrices for each joint,
    // and returns the end-effector transformation matrix.
    cv::Mat forward(const std::vector<double>& joint_angles);
    
    // get_ts: returns the list of transformation matrices for each joint.
    std::vector<cv::Mat> get_ts() const;
    
private:
    // Helper function to compute a 4x4 DH transformation matrix for one joint.
    cv::Mat compute_dh_transform(const DHParams& dh, double joint_angle) const;
    
    std::vector<DHParams> dh_params_;
    std::vector<cv::Mat> ts_; // Cached transformation matrices after forward() is called.
};