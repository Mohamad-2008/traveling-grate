#include "AnalyseGrate.hpp"
#include "AnalyseCropped.hpp"
AnalyseGrate::AnalyseGrate() : plate_detector_("F:/foolad/model/mod1.xml") {}
AnalyseCropped cropped_detector("F:/foolad/model/mod2.xml");



using namespace cv;
using namespace std;
bool AnalyseGrate::shouldProcessFrame() {
    if (allowProcessing_) {
        frameCounter_ = 0;
        std::cout << "Processing allowed (TRUE)\n";
        return true;
    }

    frameCounter_++;
    std::cout << "Processing skipped (" << frameCounter_ << "/" << skipFrames_ << ")\n";
    if (frameCounter_ >= skipFrames_) {
        allowProcessing_ = true;
    }
    return false;
}

cv::Mat AnalyseGrate::doProcess(const cv::Mat& frame) {
    if (!shouldProcessFrame()) {
        return frame.clone();
    }

    cv::Mat preprocessed = preprocess(frame);
    

    std::vector<Detection> detections = plate_detector_.detect(preprocessed);


    cv::Mat original_frame = preprocessed.clone();

    // Draw red line at row 650
    cv::line(preprocessed, cv::Point(0, 650), cv::Point(preprocessed.cols, 700), cv::Scalar(0, 0, 255), 2);

    // Draw red line at row 850
    cv::line(preprocessed, cv::Point(0, 750), cv::Point(preprocessed.cols, 800), cv::Scalar(0, 0, 255), 2);
        

    for (const auto& det : detections) {
        if (det.confidence > 0.85f) {
            cv::rectangle(preprocessed, det.box, cv::Scalar(0, 255, 0), 2);
            std::string label = plate_detector_.get_names()[det.class_id];
            cv::putText(preprocessed, label, det.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
        }
    }
    auto validDetection = findValidGrate(detections);


    CroppedDetectionInfo savedDetections;


    if (!validDetection.empty()) {
        int index = 0;
        for (const auto& det : validDetection) {
            // Crop the region from original_frame
            cv::Rect roi = det.box & cv::Rect(0, 0, original_frame.cols, original_frame.rows);
            if (roi.area() > 0) {
                cv::Mat cropped = original_frame(roi);
                savedDetections.croppedImage = cropped;

                // شناسایی با مدل دوم
                std::vector<Detection> cropped_detections = cropped_detector.detect(cropped);
                auto cropped_class_names = cropped_detector.get_names();
                auto main_class_names = plate_detector_.get_names(); // نام کلاس‌های مدل اول

                // نمایش کلاس مدل اول
                cout << "Main model class: " << main_class_names[det.class_id] << endl;

                // نمایش کلاس‌های مدل دوم
                if (!cropped_detections.empty()) {
                    for (const auto& cropped_det : cropped_detections) {
                        cout << "Cropped model class: " << cropped_class_names[cropped_det.class_id]
                            << " (confidence: " << cropped_det.confidence << ")" << endl;
                    }
                }
                else {
                    cout << "No detections found in cropped image" << endl;
                }
            }
            else {
                std::cout << "Invalid ROI for detection at index " << index << ": " << roi << "\n";
            }
            index++;
        }
        std::cout << "Found at least 3 valid detections.\n";
        allowProcessing_ = false;
    }




    return preprocessed;
}

cv::Mat AnalyseGrate::preprocess(const cv::Mat& frame) {
    cv::Mat masked_frame = frame.clone();
    cv::rectangle(masked_frame, cv::Rect(0, 0, masked_frame.cols, 400), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::rectangle(masked_frame, cv::Rect(0, masked_frame.rows - 1100, masked_frame.cols, 1100), cv::Scalar(0, 0, 0), cv::FILLED);
    return masked_frame;
}

std::vector<cv::Rect> AnalyseGrate::detectGrates(const cv::Mat& frame) {
    std::vector<cv::Rect> grates;
    // Placeholder: Replace with actual detection logic
    grates.emplace_back(100, 100, 150, 150);
    return grates;
}

std::vector<Detection> AnalyseGrate::findValidGrate(std::vector<Detection> detections) {
    std::vector<Detection> validDetections;

    for (const auto& det : detections) {
        int y = det.box.y + det.box.height / 2;  // Center Y of box

        if (det.class_id == 0) {
            if (y >= 650 && y <= 750) {
                std::cout << "crossed ->" << det.class_id << std::endl;
                validDetections.push_back(det);
            }
        }
    }

    // If fewer than 3 valid detections, clear the vector (return empty)
    if (validDetections.size() < 3) {
        validDetections.clear();
    }

    return validDetections;
}
void AnalyseGrate::trackDetection(const cv::Mat& frame, const cv::Rect& grate) {
    std::cout << "Tracking Grate at: " << grate << "\n";
}

void AnalyseGrate::detectHole(const cv::Mat& frame, const cv::Rect& grate) {
    std::cout << "Detecting holes in Grate at: " << grate << "\n";
}