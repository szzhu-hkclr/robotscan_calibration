#include "json_utils.hpp"
#include <fstream>

Config load_config(const std::string& config_path) {
    Config config;
    std::ifstream f(config_path);
    nlohmann::json j;
    f >> j;
    
    config.image_folder = j["image_folder"];
    config.pose_folder = j["pose_folder"];
    config.square_size = j["square_size"];
    config.pattern_size = j["pattern_size"].get<std::vector<int>>();
    config.dh_params = j["dh_params"].get<std::vector<std::vector<double>>>();
    config.group_lists = j["group_lists"].get<std::vector<std::string>>();
    
    return config;
}