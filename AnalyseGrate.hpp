#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include "GrateDetector.hpp"
#include "detection.hpp"

struct CroppedDetectionInfo {
    float score;                         // درصد گرفتگی
    cv::Mat croppedImage;                // تصویر برش‌خورده
    std::vector<cv::Rect> relatedBoxes;  // خرابی ها
    std::string label;                   // نام کلاس یا هر توضیح دیگر
};


class AnalyseGrate {
public:
    AnalyseGrate();
    bool shouldProcessFrame();
    cv::Mat doProcess(const cv::Mat& frame);
    std::vector<cv::Rect> detectGrates(const cv::Mat& frame);
    std::vector<Detection> findValidGrate(std::vector<Detection>);
    void trackDetection(const cv::Mat& frame, const cv::Rect& grate);
    void detectHole(const cv::Mat& frame, const cv::Rect& grate);

private:
    cv::Mat preprocess(const cv::Mat& frame);
    PlateDetector plate_detector_;
    int frameCounter_ = 0;
    const int skipFrames_ = 80;
    bool allowProcessing_ = true;

};