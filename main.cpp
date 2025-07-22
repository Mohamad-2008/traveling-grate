#include "AnalyseGrate.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap("F:/foolad/T02/1.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video\n";
        return -1;
    }

    AnalyseGrate analyzer;
    cv::Mat frame;

    while (cap.read(frame)) {
        if (frame.empty()) {
            std::cerr << "Empty frame received\n";
            break;
        }
        cv::Mat resized_frame;
        cv::Mat processed_frame = analyzer.doProcess(frame);
        cv::resize(processed_frame, resized_frame, cv::Size(), 1.0 / 3.0, 1.0 / 3.0); // کاهش اندازه به یک‌سوم
        cv::imshow("Processed Frame", resized_frame);
        if (cv::waitKey(1) == 27) break; // ESC to quit
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}