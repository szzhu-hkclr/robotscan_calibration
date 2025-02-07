#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct ChessboardResult {
    std::vector<std::vector<cv::Point2f>> image_points;
    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<int> valid_indices;
};

std::vector<cv::Mat> read_images(const std::string& folder);
ChessboardResult find_chessboard_corners(const std::vector<cv::Mat>& images, 
                                        cv::Size pattern_size, 
                                        float square_size, 
                                        bool show_corners = false);