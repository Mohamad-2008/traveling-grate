#pragma once
#include <opencv2/opencv.hpp>

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};