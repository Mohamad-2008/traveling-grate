#include "AnalyseCropped.hpp"
#include <stdexcept>
#include <iostream>
#include <opencv2/imgproc.hpp>

void AnalyseCropped::drawDetections(cv::Mat& image, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);

        std::string label = class_names_[det.class_id] + " " + cv::format("%.2f", det.confidence);
        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int top = std::max(det.box.y, labelSize.height);
        cv::rectangle(image, cv::Point(det.box.x, top - labelSize.height),
            cv::Point(det.box.x + labelSize.width, top + baseLine),
            cv::Scalar(0, 255, 0), cv::FILLED);

        cv::putText(image, label, cv::Point(det.box.x, top),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

AnalyseCropped::AnalyseCropped(const std::string& model_path) {
    try {
        core_ = ov::Core();
        auto model = core_.read_model(model_path);
        compiled_model_ = core_.compile_model(model, "AUTO", ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
        infer_request_ = compiled_model_.create_infer_request();

        // Define class names for cropped image analysis - adjust these based on your model's classes
        class_names_ = { "damaged", "slot", "damaged" };
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Failed to load cropped analysis model: " + std::string(e.what()));
    }
}

std::vector<Detection> AnalyseCropped::detect(const cv::Mat& frame) {
    const int input_w = 544, input_h = 544;
    const float CONF_THRESHOLD = 0.3f, IOU_THRESHOLD = 0.5f;
    int frame_w = frame.cols, frame_h = frame.rows;
    // Preprocess frame
    cv::Mat rgb, resized;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, { input_w, input_h });
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    // Prepare input tensor
    std::vector<float> blob(1 * 3 * input_h * input_w);
    for (int c = 0; c < 3; ++c) {
        cv::Mat channel(input_h, input_w, CV_32F, blob.data() + c * input_h * input_w);
        cv::extractChannel(resized, channel, c);
    }

    ov::Tensor input_tensor(ov::element::f32, { 1, 3, input_h, input_w }, blob.data());
    infer_request_.set_input_tensor(input_tensor);

    // Run inference
    infer_request_.infer();

    // Process output
    ov::Tensor output = infer_request_.get_output_tensor();
    auto out_shape = output.get_shape();

    // Check for expected output shape [1, 7, 6069]
    if (out_shape != ov::Shape({ 1, 7, 6069 })) {
        std::cerr << "Unexpected output shape: " << out_shape << std::endl;
        return {};
    }

    const float* out_data = output.data<const float>();
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (size_t i = 0; i < 6069; ++i) {
        float x = out_data[0 * 6069 + i]; // x_center
        float y = out_data[1 * 6069 + i]; // y_center
        float w = out_data[2 * 6069 + i]; // width
        float h = out_data[3 * 6069 + i]; // height

        float best_score = 0.0f;
        int class_id = -1;

        for (size_t c = 0; c < class_names_.size(); ++c) {
            float score = out_data[(4 + c) * 6069 + i];
            if (score > best_score) {
                best_score = score;
                class_id = c;
            }
        }

        if (best_score > CONF_THRESHOLD) {
            int left = static_cast<int>((x - w / 2.0f) * frame_w / input_w);
            int top = static_cast<int>((y - h / 2.0f) * frame_h / input_h);
            int width = static_cast<int>(w * frame_w / input_w);
            int height = static_cast<int>(h * frame_h / input_h);

            // Ensure bounding box is within image bounds
            left = std::max(0, std::min(left, frame_w - 1));
            top = std::max(0, std::min(top, frame_h - 1));
            width = std::max(1, std::min(width, frame_w - left));
            height = std::max(1, std::min(height, frame_h - top));

            boxes.emplace_back(left, top, width, height);
            confidences.push_back(best_score);
            class_ids.push_back(class_id);
        }
    }

    // Apply Non-Maximum Suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD, indices);

    // Prepare final detections
    std::vector<Detection> detections;
    for (int idx : indices) {
        detections.push_back({ boxes[idx], confidences[idx], class_ids[idx] });
    }
    cv::Mat image_with_boxes = frame.clone();
    drawDetections(image_with_boxes, detections);
    // حالا می‌توانی تصویر را ذخیره یا نمایش دهی
	imshow("Detections", image_with_boxes);
    return detections;
}

std::vector<std::string> AnalyseCropped::get_names() const {
    return class_names_;
}