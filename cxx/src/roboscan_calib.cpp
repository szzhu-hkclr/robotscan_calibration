#include "handeye_6DoF.hpp"
#include "handeye_newkin.hpp"

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <command> <config.json>" << std::endl;
        std::cerr << "Available commands: hand_eye, other" << std::endl;
        return -1;
    }

    std::string command = argv[1];
    std::string config_file = argv[2];

    try {
        Config config = load_config(config_file);

        if (command == "6dof") {
            return run6DoFHandEyeCalibration(config);
        }
        else if (command == "new") {
            return runNewKinHandEyeCalibration(config);
        }
        else {
            std::cerr << "Unknown command: " << command << std::endl;
            std::cerr << "Available commands: hand_eye, other" << std::endl;
            return -1;
        }
    }
    catch (const std::exception &ex) {
        std::cerr << "Exception occurred: " << ex.what() << std::endl;
        return -1;
    }
}