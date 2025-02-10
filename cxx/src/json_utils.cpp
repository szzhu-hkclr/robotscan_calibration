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
    for(auto& row : j["dh_params"]) {
        config.dh_params.push_back({
            row[0], row[1], row[2], row[3]
        });
    }
    config.group_lists = j["group_lists"].get<std::vector<std::string>>();
    config.tracker_pose_file = j["tracker_pose_file"];
    config.show_corners = j["show_corners"];
    config.show_projection_error = j["show_projection_error"];
    
    return config;
}