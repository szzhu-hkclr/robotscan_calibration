// main.cpp
// Compile with: g++ -std=c++14 main.cpp -o app `pkg-config --cflags --libs opencv4`

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

// Include nlohmann/json library (download json.hpp from https://github.com/nlohmann/json)
#include "json.hpp"

using json = nlohmann::json;
using namespace cv;
using namespace std;

//------------------------------------------------------------------------------
// Helper function: converts a JSON array (2D) into a cv::Mat (type CV_64F)
cv::Mat jsonToMat(const json &j) {
    int rows = j.size();
    int cols = j[0].size();
    cv::Mat mat(rows, cols, CV_64F);
    for (int i = 0; i < rows; i++) {
        for (int col = 0; col < cols; col++) {
            mat.at<double>(i, col) = j[i][col].get<double>();
        }
    }
    return mat;
}

//------------------------------------------------------------------------------
// Configuration structure to hold configurable parameters.
struct Config {
    cv::Mat dh_params;
    cv::Mat dh_params_new;
    std::string image_files_pattern;
    std::string robotfile_path;
    cv::Mat intrinsic_matrix;
    cv::Mat end_T_cam_ori;
    double square_size;
    cv::Size pattern_size;
};

// Load configuration from JSON file.
Config loadConfig(const std::string &filename) {
    Config config;
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "Could not open config file: " << filename << std::endl;
        exit(-1);
    }
    json j;
    ifs >> j;
    config.dh_params = jsonToMat(j["dh_params"]);
    config.dh_params_new = jsonToMat(j["dh_params_new"]);
    config.image_files_pattern = j["image_files"].get<std::string>();
    config.robotfile_path = j["robotfile_path"].get<std::string>();
    config.intrinsic_matrix = jsonToMat(j["intrinsic_matrix"]);
    config.end_T_cam_ori = jsonToMat(j["end_T_cam_ori"]);
    config.square_size = j["square_size"].get<double>();

    // pattern_size provided as an array (e.g. [11, 8]); here the first element is width (columns)
    int width = j["pattern_size"][0].get<int>();
    int height = j["pattern_size"][1].get<int>();
    config.pattern_size = cv::Size(width, height);
    return config;
}

//------------------------------------------------------------------------------
// Find chessboard corners in a set of images.
// If showCorners is true then each found image is displayed.
vector<vector<Point2f>> findChessboardCornersConfigured(const vector<Mat> &images, Size pattern_size, bool showCorners) {
    vector<vector<Point2f>> allCorners;
    
    cout << "Finding corners..." << endl;
    for (size_t i = 0; i < images.size(); i++) {
        Mat gray;
        cvtColor(images[i], gray, COLOR_BGR2GRAY);
        vector<Point2f> corners;
        bool found = findChessboardCorners(gray, pattern_size, corners);

        if (found) {
            // refine corner positions
            TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001);
            cornerSubPix(gray, corners, Size(10, 10), Size(-1, -1), criteria);
            allCorners.push_back(corners);

            // draw and optionally display the detected corners
            Mat drawImg = images[i].clone();
            drawChessboardCorners(drawImg, pattern_size, corners, found);
            if (showCorners) {
                string windowName = "Detected corner in image: " + to_string(i);
                imshow(windowName, drawImg);
                waitKey(0);
                destroyWindow(windowName);
            }
        } else {
            cout << "No chessboard found in image: " << i << endl;
        }
    }
    return allCorners;
}

//------------------------------------------------------------------------------
// Compute camera poses from chessboard corners using solvePnP.
// Returns a vector of 4x4 transformation matrices.
vector<Mat> computeCameraPoses(const vector<vector<Point2f>> &corners, Size pattern_size,
                               double square_size, const Mat &intrinsic_matrix, const Mat &distCoeffs) {
    // Prepare object points (in 3D) based on the known chessboard pattern.
    vector<Point3f> objectPoints;
    // Note: iterate rows then cols. (You may need to check ordering with your calibration.)
    for (int i = 0; i < pattern_size.height; i++) {
        for (int j = 0; j < pattern_size.width; j++) {
            objectPoints.push_back(Point3f(j * square_size, i * square_size, 0));
        }
    }
    
    vector<Mat> poses;
    for (size_t i = 0; i < corners.size(); i++) {
        Mat rvec, tvec;
        bool ok = solvePnP(objectPoints, corners[i], intrinsic_matrix, distCoeffs, rvec, tvec);
        if (!ok)
            continue;
        Mat R;
        Rodrigues(rvec, R);

        Mat pose = Mat::eye(4, 4, CV_64F);
        // Copy rotation matrix
        R.copyTo(pose(Rect(0, 0, 3, 3)));
        // Copy translation vector
        for (int r = 0; r < 3; r++) {
            pose.at<double>(r, 3) = tvec.at<double>(r, 0);
        }
        poses.push_back(pose);
    }
    return poses;
}

//------------------------------------------------------------------------------
// Convert a rotation matrix to Euler angles (in degrees).
Vec3d rotationMatrixToEulerAngles(const Mat &R) {
    double sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) +
                     R.at<double>(1, 0) * R.at<double>(1, 0));
    bool singular = sy < 1e-6;
    double x, y, z;
    if (!singular) {
        x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = atan2(-R.at<double>(2, 0), sy);
        z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    } else {
        x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }
    return Vec3d(x, y, z) * (180.0 / CV_PI);
}

//------------------------------------------------------------------------------
// Get angular error (average absolute difference per axis in degrees)
double getAngularError(const Mat &R_gt, const Mat &R_est) {
    Vec3d euler_gt = rotationMatrixToEulerAngles(R_gt);
    Vec3d euler_est = rotationMatrixToEulerAngles(R_est);
    double error = (fabs(euler_gt[0] - euler_est[0]) +
                    fabs(euler_gt[1] - euler_est[1]) +
                    fabs(euler_gt[2] - euler_est[2])) / 3.0;
    return error;
}

//------------------------------------------------------------------------------
// Compute transformation difference between the estimated and ground-truth SE(3) transforms.
pair<double, double> computeTransformationDiff(const Mat &R_est, const Mat &t_est,
                                                 const Mat &R_gt, const Mat &t_gt) {
    double rot_error = getAngularError(R_gt, R_est);
    Mat diff = t_gt - t_est;
    double trans_error = norm(diff);
    return make_pair(rot_error, trans_error);
}

//------------------------------------------------------------------------------
// Dummy robot kinematics class.
// Replace these methods with your actual kinematics implementation.
class RobotSerial {
public:
    RobotSerial(const Mat &dh_params) : dh_params_(dh_params) {
        // Dummy: set end_frame_ and ts_ as identities.
        end_frame_ = Mat::eye(4, 4, CV_64F);
        ts_.resize(6, Mat::eye(4, 4, CV_64F));
    }
    // Dummy forward kinematics: returns a 4x4 identity.
    Mat forward(const vector<double> &joint_state) {
        // Here you would use DH parameters and the joint state to compute the forward kinematics.
        return Mat::eye(4, 4, CV_64F);
    }
    Mat getEndFrame() const { return end_frame_; }
    vector<Mat> getTs() const { return ts_; }
private:
    Mat dh_params_;
    Mat end_frame_;  // For example: final transformation from the robot base.
    vector<Mat> ts_; // Dummy list of joint/link transforms.
};

//------------------------------------------------------------------------------
// Main
int main() {
    // Load configuration (make sure a "evaluate.json" file exists in the working directory)
    std::string configFilename = "evaluate.json";
    Config config = loadConfig(configFilename);

    // Use OpenCV's glob to list images with the pattern specified in the JSON config.
    vector<cv::String> image_files;
    cv::glob(config.image_files_pattern, image_files, false);
    if (image_files.empty()) {
        cerr << "No images found with pattern: " << config.image_files_pattern << endl;
        return -1;
    }
    
    vector<Mat> images;
    for (const auto &file : image_files) {
        Mat img = imread(file);
        if (img.empty()) {
            cerr << "Could not read image: " << file << endl;
            continue;
        }
        images.push_back(img);
    }
    
    // For demonstration, we use an empty distortion vector.
    Mat distCoeffs = Mat::zeros(1, 5, CV_64F);
    
    bool ShowCorners = false;  // You can add this option to the JSON config if desired.
    
    // 1. Chessboard detection & corner refinement.
    vector<vector<Point2f>> chessboard_corners = 
        findChessboardCornersConfigured(images, config.pattern_size, ShowCorners);
    
    // 2. Compute camera poses from the detected corners.
    vector<Mat> cam_T_chesss = computeCameraPoses(chessboard_corners,
                                                  config.pattern_size,
                                                  config.square_size,
                                                  config.intrinsic_matrix, distCoeffs);
    if (cam_T_chesss.empty()) {
        cerr << "No camera poses computed!" << endl;
        return -1;
    }
    Mat cam1_T_chess = cam_T_chesss[0];
    
    // NOTE:
    // In the original Python code routines point cloud processing via Open3D.
    // In this example, that portion is omitted – you can integrate a library such as PCL if needed.

    // 3. Read robot state data from file.
    vector<vector<double>> robot_states;
    ifstream robotFile(config.robotfile_path);
    if (!robotFile.is_open()) {
        cerr << "Could not open robot file: " << config.robotfile_path << endl;
    } else {
        string line;
        while (getline(robotFile, line)) {
            if (line.empty())
                continue;
            istringstream iss(line);
            vector<double> state;
            double value;
            // Assuming values are whitespace separated.
            while (iss >> value)
                state.push_back(value);
            robot_states.push_back(state);
        }
    }
    
    // 4. 6-DoF Kinematics using the first DH parameters.
    RobotSerial robot_ori(config.dh_params);
    vector<Mat> base_T_camidxs;
    for (size_t i = 0; i < robot_states.size(); i++) {
        Mat f = robot_ori.forward(robot_states[i]);
        // In the Python code the transformation is computed by multiplying the robot's end-frame 
        // with a constant transformation (end_T_cam_ori). Here we perform the same.
        Mat base_T_camidx = robot_ori.getEndFrame() * config.end_T_cam_ori;
        base_T_camidxs.push_back(base_T_camidx);
    }
    
    // (For demonstration, we assume the first measurement as the reference.)
    Mat base_T_cam1 = base_T_camidxs[0];
    
    // 5. 6-DoF calibration using new robot kinematics (with dh_params_new).
    RobotSerial robot_new(config.dh_params_new);
    
    // In the Python code, tracker_T_3s is loaded from a .npy file.
    // Here we create a dummy vector (one 4x4 identity per robot state).
    vector<Mat> tracker_T_3s(robot_states.size(), Mat::eye(4, 4, CV_64F));
    
    vector<Mat> tracker_T_camidxs;
    for (size_t idx = 0; idx < robot_states.size(); idx++) {
        vector<double> state = robot_states[idx];
        Mat f = robot_new.forward(state);
        // For demonstration, we simulate a chain multiplication (Link3TEnd_i) as identity.
        Mat Link3TEnd_i = Mat::eye(4, 4, CV_64F);
        Mat tracker_T_camidx = tracker_T_3s[idx] * Link3TEnd_i * config.end_T_cam_ori;
        tracker_T_camidxs.push_back(tracker_T_camidx);
    }
    
    // 6. Compute transformation differences (error metrics)
    // Here we compare (dummy) transformations.
    vector<double> errors_R_K6DoF, errors_t_K6DoF;
    vector<double> errors_R_K3DoF, errors_t_K3DoF;
    
    // For demonstration, we loop over the computed camera poses.
    for (size_t i = 0; i < cam_T_chesss.size(); i++) {
        // Using cam1_T_chess as the ground truth
        Mat R_est = cam1_T_chess(Rect(0, 0, 3, 3));
        Mat t_est = cam1_T_chess(Range(0, 3), Range(3, 4));
        Mat R_gt = cam1_T_chess(Rect(0, 0, 3, 3));
        Mat t_gt = cam1_T_chess(Range(0, 3), Range(3, 4));
        auto diff = computeTransformationDiff(R_est, t_est, R_gt, t_gt);
        errors_R_K6DoF.push_back(diff.first);
        errors_t_K6DoF.push_back(diff.second * 1000.0);
        errors_R_K3DoF.push_back(diff.first);
        errors_t_K3DoF.push_back(diff.second * 1000.0);
    }
    
    // 7. Print error metrics.
    cout << "6DoF Rotation Errors (deg):" << endl;
    for (auto e : errors_R_K6DoF)
        cout << e << " ";
    cout << "\n6DoF Translation Errors (mm):" << endl;
    for (auto e : errors_t_K6DoF)
        cout << e << " ";
    cout << "\n3DoF Rotation Errors (deg):" << endl;
    for (auto e : errors_R_K3DoF)
        cout << e << " ";
    cout << "\n3DoF Translation Errors (mm):" << endl;
    for (auto e : errors_t_K3DoF)
        cout << e << " ";
    cout << endl;
    
    return 0;
}