#include "GrateDetector.hpp"
#include <stdexcept>

PlateDetector::PlateDetector(const std::string& model_path) {
    try {
        core_ = ov::Core();
        auto model = core_.read_model(model_path);
        compiled_model_ = core_.compile_model(model, "AUTO", ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
        infer_request_ = compiled_model_.create_infer_request();
        class_names_ = { "correct_grate-plate", "broken_grate-plate", "middle" };
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Failed to load plate detector model: " + std::string(e.what()));
    }
}

std::vector<Detection> PlateDetector::detect(const cv::Mat& frame) {
    const int input_w = 640, input_h = 640;
    const float CONF_THRESHOLD = 0.85f, IOU_THRESHOLD = 0.5f;
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
	//std::cout << "Running inference on input tensor of shape: " << input_tensor.get_shape() << std::endl;
    infer_request_.infer();

    // Process output
    ov::Tensor output = infer_request_.get_output_tensor();
    auto out_shape = output.get_shape();
    if (out_shape != ov::Shape({ 1, 7, 8400 })) {
        std::cerr << "Unexpected output shape: " << out_shape << std::endl;
        return {};
    }

    const float* out_data = output.data<const float>();
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (size_t i = 0; i < 8400; ++i) {
        float x = out_data[0 * 8400 + i]; // x_center
        float y = out_data[1 * 8400 + i]; // y_center
        float w = out_data[2 * 8400 + i]; // width
        float h = out_data[3 * 8400 + i]; // height

        float best_score = 0.0f;
        int class_id = -1;
        for (size_t c = 0; c < class_names_.size(); ++c) {
            float score = out_data[(4 + c) * 8400 + i];
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
            boxes.emplace_back(left, top, width, height);
            confidences.push_back(best_score);
            class_ids.push_back(class_id);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD, indices);

    std::vector<Detection> detections;
    for (int idx : indices) {
        detections.push_back({ boxes[idx], confidences[idx], class_ids[idx] });
    }
    return detections;
}

std::vector<std::string> PlateDetector::get_names() const {
    return class_names_;
}