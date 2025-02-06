// main.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

// Include the nlohmann JSON header (download from https://github.com/nlohmann/json)
#include "json.hpp"

// Use the JSON namespace for convenience
using json = nlohmann::json;
using namespace std;

// ---------------------------
// A dummy RobotSerial class.
// In an actual application, replace this with your own implementation.
class RobotSerial {
public:
    // Ts holds the transformation for each link
    // For simplicity, we use a vector of 4x4 cv::Mat matrices.
    std::vector<cv::Mat> ts;

    // The constructor receives the DH parameters (each row: [d, a, alpha, theta])
    RobotSerial(const std::vector<std::vector<double>>& dh_params) {
        // For simplicity, we store an identity for every joint.
        // Replace this with your DH forward kinematics calculations.
        size_t n = dh_params.size();
        ts.resize(n);
        for (size_t i = 0; i < n; i++) {
            ts[i] = cv::Mat::eye(4, 4, CV_64F);
        }
    }

    // forward calculates the forward kinematics for a given joint vector.
    // For this dummy function, we simply return an identity 4x4 matrix.
    cv::Mat forward(const std::vector<double>& joint_angles) {
        // In a real implementation, use the DK formula with dh_params and joint_angles.
        // For demonstration, we update ts based on joint_angles in some trivial way.
        // Here, we simply update the last joint transform with a translation.
        cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
        // For example, assume joint_angles[0] adds a translation in x.
        T.at<double>(0, 3) = joint_angles.empty() ? 0 : joint_angles[0];
        return T;
    }
};

// ---------------------------
// Function to read JSON configuration
bool readConfig(const std::string& filename, json &config) {
    std::ifstream inFile(filename);
    if (!inFile.is_open()){
        std::cerr << "Error: Cannot open configuration file " << filename << std::endl;
        return false;
    }
    inFile >> config;
    return true;
}

// ---------------------------
// Finds chessboard corners in a set of images
std::vector<std::vector<cv::Point2f>> findChessboardCornersCustom(
    const std::vector<cv::Mat>& images,
    cv::Size patternSize,
    bool showCorners,
    std::vector<int>& indices
) {
    std::vector<std::vector<cv::Point2f>> chessboardCorners;
    int idx = 0;
    std::cout << "Finding corners..." << std::endl;
    for (const auto& img : images) {
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, patternSize, corners);
        if (found) {
            cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);
            cv::cornerSubPix(gray, corners, cv::Size(10, 10), cv::Size(-1, -1), criteria);
            chessboardCorners.push_back(corners);

            // Draw the corners on a copy of the image
            cv::Mat drawImg = img.clone();
            cv::drawChessboardCorners(drawImg, patternSize, corners, found);
            if (showCorners) {
                cv::imshow("Detected corners in image " + std::to_string(idx), drawImg);
                cv::waitKey(0);
            }
            indices.push_back(idx);
        } else {
            std::cout << "No chessboard found in image: " << idx << std::endl;
        }
        idx++;
    }
    return chessboardCorners;
}

// ---------------------------
// Calculate camera intrinsics and distortion coefficients.
void calculateIntrinsics(
    const std::vector<std::vector<cv::Point2f>>& imagePoints,
    const std::vector<int>& indices,
    cv::Size patternSize,
    double squareSize,
    cv::Size imageSize,
    cv::Mat& cameraMatrix,
    cv::Mat& distCoeffs,
    double& reprojectionError,
    bool showProjectError = false
) {
    // Prepare object points: same for all images
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<cv::Point3f> objp;
    for (int i = 0; i < patternSize.height; i++) {
        for (int j = 0; j < patternSize.width; j++) {
            objp.push_back(cv::Point3f(j * squareSize, i * squareSize, 0));
        }
    }
    for (size_t i = 0; i < indices.size(); i++) {
        objectPoints.push_back(objp);
    }

    std::vector<cv::Mat> rvecs, tvecs;
    int flags = 0;
    double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags);
    std::cout << "Calibration RMS error: " << rms << std::endl;

    // Compute reprojection error.
    double totalError = 0;
    int totalPoints = 0;
    for (size_t i = 0; i < objectPoints.size(); i++) {
        std::vector<cv::Point2f> projectedPoints;
        cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, projectedPoints);
        double err = cv::norm(imagePoints[i], projectedPoints, cv::NORM_L2);
        int n = objectPoints[i].size();
        totalError += err * err;
        totalPoints += n;
        if (showProjectError) {
            std::cout << "Image " << i << " error: " << err / n << std::endl;
        }
    }
    reprojectionError = std::sqrt(totalError / totalPoints);
    std::cout << "Mean reprojection error: " << reprojectionError << std::endl;
}

// ---------------------------
// Compute camera poses using solvePnP
void computeCameraPoses(
    const std::vector<std::vector<cv::Point2f>>& chessboardCorners,
    cv::Size patternSize,
    double squareSize,
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs,
    std::vector<cv::Mat>& rvecs,
    std::vector<cv::Mat>& tvecs
) {
    // Prepare object points.
    std::vector<cv::Point3f> objectPoints;
    for (int i = 0; i < patternSize.height; i++) {
        for (int j = 0; j < patternSize.width; j++) {
            objectPoints.push_back(cv::Point3f(j * squareSize, i * squareSize, 0));
        }
    }

    for (size_t i = 0; i < chessboardCorners.size(); i++) {
        cv::Mat rvec, tvec;
        bool solved = cv::solvePnP(objectPoints, chessboardCorners[i], cameraMatrix, distCoeffs, rvec, tvec);
        if (!solved) {
            std::cerr << "solvePnP failed for image index " << i << std::endl;
        }
        rvecs.push_back(rvec);
        tvecs.push_back(tvec);
    }
}

// ---------------------------
// Read joints (poses) from file. Each line is a space‐separated list.
std::vector<std::vector<double>> readJoints(const std::string &filename) {
    std::vector<std::vector<double>> joints;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Failed to open joint file: " << filename << std::endl;
        return joints;
    }
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        double value;
        std::vector<double> pose;
        while (iss >> value) {
            pose.push_back(value);
        }
        if (!pose.empty())
            joints.push_back(pose);
    }
    return joints;
}

// ---------------------------
// Main function
int main() {
    // Read configuration from handeye.json
    json config;
    if (!readConfig("handeye.json", config)) {
        return -1;
    }
    
    // Configurable parameters from json
    // Expect keys: "dh_params", "num_group", "image_folder", "pose_folder", "square_size", "pattern_size", "group_lists"
    std::vector<std::vector<double>> dh_params = config["dh_params"].get<std::vector<std::vector<double>>>();
    int num_group = config["num_group"].get<int>();
    std::string image_folder = config["image_folder"].get<std::string>();
    std::string pose_folder = config["pose_folder"].get<std::string>();
    double square_size = config["square_size"].get<double>();
    // Expect pattern_size to be a list of two integers [width, height]
    std::vector<int> pattern_size_vec = config["pattern_size"].get<std::vector<int>>();
    cv::Size patternSize(pattern_size_vec[0], pattern_size_vec[1]);
    std::vector<std::string> group_lists = config["group_lists"].get<std::vector<std::string>>();

    // Optionally, you can also control if corners or project error are shown.
    bool showCorners = config.value("show_corners", false);
    bool showProjectError = config.value("show_project_error", false);

    // Read images from each group folder (assumes .png images)
    std::vector<cv::Mat> images;
    for (const auto& group : group_lists) {
        std::string groupPath = image_folder + "/" + group;
        // Use cv::glob to find all matching image files
        std::vector<cv::String> imageFiles;
        cv::glob(groupPath + "/*.png", imageFiles, false);
        for (size_t i = 0; i < imageFiles.size(); i++) {
            cv::Mat img = cv::imread(imageFiles[i]);
            if (!img.empty()) {
                images.push_back(img);
            } else {
                std::cerr << "Failed to read image: " << imageFiles[i] << std::endl;
            }
        }
    }
    
    if (images.empty()) {
        std::cerr << "No images found. Exiting." << std::endl;
        return -1;
    }
    
    // Assuming all images are the same size, get the size from the first image.
    cv::Size imageSize = images[0].size();
    
    // Find chessboard corners
    std::vector<int> indices;
    std::vector<std::vector<cv::Point2f>> chessboardCorners = 
        findChessboardCornersCustom(images, patternSize, showCorners, indices);
    
    if(chessboardCorners.empty()){
        std::cerr << "No valid chessboard corners found. Exiting." << std::endl;
        return -1;
    }
    
    // Camera calibration
    cv::Mat cameraMatrix, distCoeffs;
    double reprojectionError = 0.0;
    calculateIntrinsics(chessboardCorners, indices, patternSize, square_size, imageSize, cameraMatrix, distCoeffs, reprojectionError, showProjectError);
    
    // Compute camera extrinsics (poses)
    std::vector<cv::Mat> rvecs, tvecs;
    computeCameraPoses(chessboardCorners, patternSize, square_size, cameraMatrix, distCoeffs, rvecs, tvecs);
    
    // For hand-eye calibration, we need to compute transformation T_end2base from robot poses.
    // These containers will store rotation (3x3) and translation (3x1).
    std::vector<cv::Mat> REnd2Base;
    std::vector<cv::Mat> TEnd2Base;
    
    // Load tracker_T_3 (assume a .npy file loaded externally)
    // For simplicity, assume a dummy transformation for every group.
    // In practice one can use cnpy (https://github.com/rogersce/cnpy) or another loader.
    // Here we assume tracker_T_3s is a vector (one per group) of 4x4 cv::Mat.
    std::vector<cv::Mat> tracker_T_3s;
    for (int i = 0; i < num_group; i++) {
        tracker_T_3s.push_back(cv::Mat::eye(4, 4, CV_64F)); // dummy identity
    }
    
    // Initialize robot (using configured dh_params)
    RobotSerial robot(dh_params);
    
    // For each group, load the joint poses and compute the corresponding transformations.
    for (size_t i = 0; i < group_lists.size(); i++) {
        std::string pose_file = pose_folder + "/" + group_lists[i] + ".txt";
        std::vector<std::vector<double>> pose_group = readJoints(pose_file);
        for (const auto& pose : pose_group) {
            cv::Mat T_robot = robot.forward(pose);
            // Compute the transformation from link3 to end effector.
            cv::Mat Link3TEnd = cv::Mat::eye(4, 4, CV_64F);
            // For joints [3, 4, 5] (0-indexed) multiply corresponding transformation matrices.
            for (int j = 3; j <= 5; j++) {
                // Assuming robot.ts[j] is a full 4x4 transform.
                Link3TEnd = Link3TEnd * robot.ts[j];
            }
            // Compute final transformation T_end
            cv::Mat T_final = tracker_T_3s[i] * Link3TEnd;
            // Extract rotation and translation from T_final
            cv::Mat R = T_final(cv::Rect(0, 0, 3, 3)).clone();
            cv::Mat t = T_final(cv::Rect(3, 0, 1, 3)).clone();
            REnd2Base.push_back(R);
            TEnd2Base.push_back(t);
        }
    }
    
    // Hand-eye calibration using OpenCV's calibrateHandEye.
    // Make sure you use a recent OpenCV version (>=4.x) that supports the function.
    cv::Mat R_cam2gripper, t_cam2gripper;
    // For hand eye calibration, rvecs and tvecs from camera and REnd2Base and TEnd2Base from robot are used.
    // Here we use method = cv::CALIB_HAND_EYE_TSAI.
    cv::calibrateHandEye(REnd2Base, TEnd2Base, rvecs, tvecs, R_cam2gripper, t_cam2gripper, cv::CALIB_HAND_EYE_TSAI);
    
    // Compose the final transformation matrix end_T_cam
    cv::Mat end_T_cam = cv::Mat::eye(4, 4, CV_64F);
    R_cam2gripper.copyTo(end_T_cam(cv::Rect(0,0,3,3)));
    // t_cam2gripper is a 3x1 matrix.
    t_cam2gripper.copyTo(end_T_cam(cv::Rect(3,0,1,3)));
    
    std::cout << "end_T_cam: " << std::endl << end_T_cam << std::endl;
    
    return 0;
}