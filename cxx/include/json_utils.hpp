#pragma once
#include "json.hpp"
#include "RobotSerial.hpp"
#include <string>
#include <vector>

struct Config {
    std::string image_folder;
    std::string pose_folder;
    float square_size;
    std::vector<int> pattern_size;
    std::vector<DHParams> dh_params;
    std::vector<DHParams> dh_params_new;
    std::vector<std::string> group_lists;
    std::string tracker_pose_file;
    std::string image_files;
    std::string robotfile_path;
    std::vector<std::vector<double>> intrinsic_matrix;
    std::vector<std::vector<double>> end_T_cam_ori;
    
    bool show_corners;
    bool show_projection_error;
};

Config load_config(const std::string& config_path);