// main.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <nlohmann/json.hpp>

// For convenience
using json = nlohmann::json;

// ---------------- RobotSerial CLASS ----------------
class RobotSerial {
public:
    // dh_params: vector of {d, a, alpha, theta_offset}
    RobotSerial(const std::vector<std::vector<double>>& dh_params)
        : dh_params_(dh_params) {}

    // forward kinematics: joint_angles must have size equal to dh_params_.size()
    cv::Mat forward(const std::vector<double>& joint_angles) {
        cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
        for (size_t i = 0; i < dh_params_.size(); i++) {
            double d          = dh_params_[i][0];
            double a          = dh_params_[i][1];
            double alpha      = dh_params_[i][2];
            double theta_off  = dh_params_[i][3];
            double theta = joint_angles[i] + theta_off;
            double cos_theta = cos(theta);
            double sin_theta = sin(theta);
            double cos_alpha = cos(alpha);
            double sin_alpha = sin(alpha);
            cv::Mat A = (cv::Mat_<double>(4,4) <<
                cos_theta, -sin_theta * cos_alpha,  sin_theta * sin_alpha, a * cos_theta,
                sin_theta,  cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta,
                0,          sin_alpha,              cos_alpha,             d,
                0,          0,                      0,                     1);
            T = T * A;
        }
        return T;
    }

private:
    std::vector<std::vector<double>> dh_params_;
};

// ---------------- HELPERS ----------------

// Read the configuration file (JSON)
bool readConfig(const std::string& config_path, json &config) {
    std::ifstream config_file(config_path);
    if (!config_file.is_open()){
        std::cerr << "Could not open config file: " << config_path << std::endl;
        return false;
    }
    config_file >> config;
    return true;
}

// Read joints from text file. Each line contains joint angles separated by whitespace (and possibly commas).
std::vector<std::vector<double>> readJoints(const std::string& filename) {
    std::vector<std::vector<double>> joints;
    std::ifstream infile(filename);
    if (!infile.is_open()){
        std::cerr << "Could not open joint file: " << filename << std::endl;
        return joints;
    }
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::vector<double> joint;
        std::string token;
        while (iss >> token) {
            // Remove any commas
            token.erase(remove(token.begin(), token.end(), ','), token.end());
            joint.push_back(std::stod(token));
        }
        if(!joint.empty()){
            joints.push_back(joint);
        }
    }
    return joints;
}

// ---------------- CV FUNCTIONS ----------------

// Find chessboard corners in the given images.
void findChessboardCorners(const std::vector<cv::Mat>& images,
                           const cv::Size& patternSize,
                           bool showCorners,
                           std::vector<std::vector<cv::Point2f>>& chessboardCorners,
                           std::vector<int>& indicesWithImg) {
    std::cout << "Finding corners..." << std::endl;
    int idx = 0;
    for (const auto& image : images) {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, patternSize, corners);
        if (found) {
            // refine corners
            cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);
            cv::cornerSubPix(gray, corners, cv::Size(10, 10), cv::Size(-1, -1), criteria);
            chessboardCorners.push_back(corners);
            indicesWithImg.push_back(idx);

            cv::Mat img_draw = image.clone();
            cv::drawChessboardCorners(img_draw, patternSize, corners, found);
            if (showCorners) {
                std::string winName = "Detected corner in image: " + std::to_string(idx);
                cv::imshow(winName, img_draw);
                cv::waitKey(0);
                cv::destroyWindow(winName);
            }
        }
        else {
            std::cout << "No chessboard found in image: " << idx << std::endl;
        }
        idx++;
    }
}

// Calculate the intrinsic camera matrix and distortion coefficients.
void calculateIntrinsics(const std::vector<std::vector<cv::Point2f>>& chessboardCorners,
                         const std::vector<int>& indicesWithImg,
                         const cv::Size& patternSize,
                         double squareSize,
                         const cv::Size& imageSize,
                         bool showProjectError,
                         cv::Mat& intrinsicMatrix,
                         cv::Mat& distCoeffs,
                         double& reprojectionError) {
    std::vector<std::vector<cv::Point3f>> objectPoints;
    // Prepare object points for the pattern
    for (size_t i = 0; i < indicesWithImg.size(); i++) {
        std::vector<cv::Point3f> objp;
        for (int j = 0; j < patternSize.height; j++){
            for (int k = 0; k < patternSize.width; k++){
                objp.push_back(cv::Point3f(k * squareSize, j * squareSize, 0));
            }
        }
        objectPoints.push_back(objp);
    }

    std::vector<cv::Mat> rvecs, tvecs;
    int flags = 0;
    double rms = cv::calibrateCamera(objectPoints, chessboardCorners, imageSize, intrinsicMatrix,
                                     distCoeffs, rvecs, tvecs, flags);
    std::cout << "Re-projection error from calibration: ";

    // Compute the reprojection error
    double totalError = 0;
    size_t totalPoints = 0;
    std::vector<double> perViewErrors;
    for (size_t i = 0; i < objectPoints.size(); i++) {
        std::vector<cv::Point2f> projectedPoints;
        cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], intrinsicMatrix, distCoeffs, projectedPoints);
        double err = cv::norm(chessboardCorners[i], projectedPoints, cv::NORM_L2);
        size_t n = objectPoints[i].size();
        perViewErrors.push_back(std::sqrt(err * err / n));
        totalError += err * err;
        totalPoints += n;
    }
    reprojectionError = std::sqrt(totalError / totalPoints);
    std::cout << reprojectionError << std::endl;
}

// Compute camera poses (rotation and translation) for each chessboard image.
void computeCameraPoses(const std::vector<std::vector<cv::Point2f>>& chessboardCorners,
                        const cv::Size& patternSize,
                        double squareSize,
                        const cv::Mat& intrinsicMatrix,
                        const cv::Mat& distCoeffs,
                        std::vector<cv::Mat>& rotations,
                        std::vector<cv::Mat>& translations,
                        bool testing = false) {
    // Create object points for one chessboard image.
    std::vector<cv::Point3f> objectPoints;
    for (int j = 0; j < patternSize.height; j++){
        for (int k = 0; k < patternSize.width; k++){
            objectPoints.push_back(cv::Point3f(k * squareSize, j * squareSize, 0));
        }
    }
    int iter = 0;
    for (const auto & corners : chessboardCorners) {
        cv::Mat rvec, tvec;
        bool success = cv::solvePnP(objectPoints, corners, intrinsicMatrix, distCoeffs, rvec, tvec);
        if (!success) {
            std::cerr << "solvePnP failed for image " << iter << std::endl;
            continue;
        }
        if (testing) {
            std::cout << "Iteration: " << iter << std::endl;
            std::cout << "rvec: " << rvec.t() << std::endl;
            std::cout << "tvec: " << tvec.t() << std::endl;
            std::cout << "-------------------" << std::endl;
        }
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        rotations.push_back(R);
        translations.push_back(tvec);
        iter++;
    }
}

// Utility: Use cv::glob to get list of filenames matching a pattern
std::vector<std::string> getFiles(const std::string& pattern) {
    std::vector<std::string> files;
    cv::glob(pattern, files, false);
    return files;
}

// ---------------- MAIN ----------------

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./handeye_6DoF handeye.json" << std::endl;
        return -1;
    }
    // Load config json
    json config;
    if (!readConfig(argv[1], config)) {
        return -1;
    }

    // Read configurable parameters from json
    std::string image_folder = config["image_folder"].get<std::string>();
    std::string pose_folder  = config["pose_folder"].get<std::string>();
    double square_size       = config["square_size"].get<double>();
    std::vector<int> pattern_vec = config["pattern_size"].get<std::vector<int>>();
    if (pattern_vec.size() < 2) {
        std::cerr << "pattern_size should be an array of two integers" << std::endl;
        return -1;
    }
    cv::Size patternSize(pattern_vec[0], pattern_vec[1]);
    bool show_project_error = config.value("show_project_error", false);
    bool show_corners       = config.value("show_corners", false);
    std::vector<std::string> group_lists = config["group_lists"].get<std::vector<std::string>>();

    // Get dh_params from JSON
    std::vector<std::vector<double>> dh_params;
    for (const auto &row : config["dh_params"]) {
        std::vector<double> rowvec = row.get<std::vector<double>>();
        dh_params.push_back(rowvec);
    }

    // Read images from each group folder
    std::vector<cv::Mat> images;
    for (const auto & group: group_lists) {
        std::string pattern = image_folder + "/" + group + "/*.png";
        std::vector<std::string> file_names = getFiles(pattern);
        for (const auto & file : file_names) {
            cv::Mat im = cv::imread(file);
            if (!im.empty()) {
                images.push_back(im);
            } else {
                std::cerr << "Could not read image: " << file << std::endl;
            }
        }
    }
    if (images.empty()) {
        std::cerr << "No images loaded." << std::endl;
        return -1;
    }

    // Find chessboard corners
    std::vector<std::vector<cv::Point2f>> chessboardCorners;
    std::vector<int> indicesWithImg;
    findChessboardCorners(images, patternSize, show_corners, chessboardCorners, indicesWithImg);
    if (chessboardCorners.empty()) {
        std::cerr << "No chessboard corners were found." << std::endl;
        return -1;
    }

    // Calculate Intrinsics
    cv::Size imageSize = images[0].size();
    cv::Mat intrinsicMatrix, distCoeffs;
    double reprojError = 0.0;
    calculateIntrinsics(chessboardCorners, indicesWithImg, patternSize, square_size,
                        imageSize, show_project_error, intrinsicMatrix, distCoeffs, reprojError);

    // Compute camera poses for each chessboard image
    std::vector<cv::Mat> RTarget2Cam, TTarget2Cam;
    computeCameraPoses(chessboardCorners, patternSize, square_size, intrinsicMatrix,
                       distCoeffs, RTarget2Cam, TTarget2Cam, false);

    // -- Robot forward kinematics to get gripper (end-effector) poses --
    RobotSerial robot(dh_params);
    std::vector<cv::Mat> REnd2Base;
    std::vector<cv::Mat> TEnd2Base;
    for (const auto & group: group_lists) {
        std::string pose_file = pose_folder + "/" + group + ".txt";
        std::vector<std::vector<double>> joints = readJoints(pose_file);
        for (const auto & joint : joints) {
            // Assume joint angles size match the dh_params size.
            cv::Mat T = robot.forward(joint);
            cv::Mat R = T(cv::Rect(0,0,3,3)).clone();
            cv::Mat t = T(cv::Rect(3,0,1,3)).clone();
            REnd2Base.push_back(R);
            TEnd2Base.push_back(t);
        }
    }
    if (REnd2Base.empty() || RTarget2Cam.empty()) {
        std::cerr << "Insufficient data for hand-eye calibration." << std::endl;
        return -1;
    }

    // Hand-Eye Calibration using OpenCV function (method=4 corresponds to Park & Martin)
    cv::Mat R_cam2gripper, t_cam2gripper;
    cv::calibrateHandEye(REnd2Base, TEnd2Base,
                         RTarget2Cam, TTarget2Cam,
                         R_cam2gripper, t_cam2gripper,
                         cv::CALIB_HAND_EYE_TSAI);

    // Construct transformation matrix
    cv::Mat end_T_cam = cv::Mat::eye(4,4,CV_64F);
    R_cam2gripper.copyTo(end_T_cam(cv::Rect(0,0,3,3)));
    t_cam2gripper.copyTo(end_T_cam(cv::Rect(3,0,1,3)));

    // Output results
    std::cout << "end_T_cam:" << std::endl << end_T_cam << std::endl;
    std::cout << "Intrinsic Matrix:" << std::endl << intrinsicMatrix << std::endl;
    std::cout << "Distortion Coefficients:" << std::endl << distCoeffs << std::endl;

    return 0;
}