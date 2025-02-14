#include "json_utils.hpp"
#include <fstream>

Config load_config(const std::string& config_path) {
    Config config;
    std::ifstream f(config_path);
    nlohmann::json j;
    f >> j;
    
    config.image_folder = j["image_folder"];
    config.pose_folder = j["pose_folder"];
    config.image_files = j["image_files"];
    config.robotfile_path = j["robotfile_path"];
    config.square_size = j["square_size"];
    config.pattern_size = j["pattern_size"].get<std::vector<int>>();
    for (auto& row : j["dh_params"]) {
        config.dh_params.push_back({
            row[0], row[1], row[2], row[3]
        });
    }
    for (auto& row : j["dh_params_new"]) {
        config.dh_params_new.push_back({
            row[0], row[1], row[2], row[3]
        });
    }

    config.group_lists = j["group_lists"].get<std::vector<std::string>>();
    config.tracker_pose_file = j["tracker_pose_file"];

    // Parse intrinsic_matrix (assumed to be 3x3)
    for (auto& row : j["intrinsic_matrix"]) {
        config.intrinsic_matrix.push_back(row.get<std::vector<double>>());
    }

    // Parse end_T_cam_ori (assumed to be 4x4)
    for (auto& row : j["end_T_cam_ori"]) {
        config.end_T_cam_ori.push_back(row.get<std::vector<double>>());
    }

    config.show_corners = j["show_corners"];
    config.show_projection_error = j["show_projection_error"];

    return config;
}