#pragma once
#include "json.hpp"
#include <string>
#include <vector>

struct Config {
    std::string image_folder;
    std::string pose_folder;
    float square_size;
    std::vector<int> pattern_size;
    std::vector<std::vector<double>> dh_params;
    std::vector<std::string> group_lists;
};

Config load_config(const std::string& config_path);