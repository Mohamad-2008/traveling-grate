struct CroppedDetectionInfo {
    float score;                         // درصد گرفتگی
    cv::Mat croppedImage;                // تصویر برش‌خورده
    std::vector<cv::Rect> relatedBoxes;  // خرابی ها
    std::string label;                   // نام کلاس یا هر توضیح دیگر
};
