#pragma once
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <vector>
#include <string>
#include "detection.hpp"

class PlateDetector {
public:
    PlateDetector(const std::string& model_path = "F:/foolad/model/mod1.xml");
    std::vector<Detection> detect(const cv::Mat& frame);
    std::vector<std::string> get_names() const;

private:
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
    std::vector<std::string> class_names_;
};