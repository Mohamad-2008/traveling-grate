#pragma once
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <vector>
#include <string>
#include "detection.hpp"

class AnalyseCropped {
public:
    AnalyseCropped(const std::string& model_path);
    std::vector<Detection> detect(const cv::Mat& frame);
    std::vector<std::string> get_names() const;
    void drawDetections(cv::Mat& image, const std::vector<Detection>& detections);


private:
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
    std::vector<std::string> class_names_;
};