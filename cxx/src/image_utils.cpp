#include "image_utils.hpp"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

std::vector<cv::Mat> read_images(const std::string& folder) {
    std::vector<cv::Mat> images;
    std::vector<fs::path> paths;
    
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.path().extension() == ".png") {
            paths.push_back(entry.path());
        }
    }
    
    std::sort(paths.begin(), paths.end());
    
    for (const auto& path : paths) {
        cv::Mat img = cv::imread(path.string());
        if (!img.empty()) {
            images.push_back(img);
        } else {
            std::cerr << "Failed to load image: " << path.string() << std::endl;
        }
    }
    return images;
}

ChessboardResult find_chessboard_corners(const std::vector<cv::Mat>& images, 
                                       cv::Size pattern_size, 
                                       float square_size, 
                                       bool show_corners) {
    ChessboardResult result;
    std::vector<cv::Point3f> object_pattern;
    
    // Build the object pattern for a chessboard (row-first)
    for (int i = 0; i < pattern_size.height; ++i) {
        for (int j = 0; j < pattern_size.width; ++j) {
            object_pattern.emplace_back(j * square_size, i * square_size, 0.0f);
        }
    }

    for (size_t i = 0; i < images.size(); ++i) {
        cv::Mat gray;
        if (images[i].channels() == 3)
            cv::cvtColor(images[i], gray, cv::COLOR_BGR2GRAY);
        else
            gray = images[i];
        
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, pattern_size, corners);
        
        if (found) {
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));
            
            result.image_points.push_back(corners);
            result.object_points.push_back(object_pattern);
            result.valid_indices.push_back(i);

            if (show_corners) {
                cv::drawChessboardCorners(images[i], pattern_size, corners, true);
                cv::imshow("Detected Corners", images[i]);
                cv::waitKey(0);
            }
        } else {
            std::cout << "No chessboard found in image: " << i << std::endl;
        }
    }
    return result;
}