<<<<<<< HEAD
cmake_minimum_required(VERSION 3.16)
project(openvino)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------
# مسیر OpenCV
# ----------------------------
set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED)

# ----------------------------
# مسیر OpenVINO - دستی
# ----------------------------
include_directories("C:/openvino/runtime/include")
link_directories("C:/openvino/runtime/lib/intel64/Release")

# ------------------------- json --------------------------------- 
include_directories(${CMAKE_SOURCE_DIR}/external/json)

# ----------------------------
# منابع پروژه
# ----------------------------
set(SOURCES
    main.cpp
    AnalyseGrate.cpp
    GrateDetector.cpp
    AnalyseCropped.cpp
    ConfigManager.cpp
)

# ----------------------------
# فایل اجرایی
# ----------------------------
add_executable(main ${SOURCES})

# ----------------------------
# لینک به کتابخانه‌ها
# ----------------------------
target_link_libraries(main
    ${OpenCV_LIBS}
    openvino)
=======
cmake_minimum_required(VERSION 3.16)
project(openvino)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------
# مسیر OpenCV
# ----------------------------
set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED)

# ----------------------------
# مسیر OpenVINO - دستی
# ----------------------------
include_directories("C:/openvino/runtime/include")
link_directories("C:/openvino/runtime/lib/intel64/Release")

# ------------------------- json --------------------------------- 
include_directories(${CMAKE_SOURCE_DIR}/external/json)

# ----------------------------
# منابع پروژه
# ----------------------------
set(SOURCES
    main.cpp
    AnalyseGrate.cpp
    GrateDetector.cpp
    AnalyseCropped.cpp
)

# ----------------------------
# فایل اجرایی
# ----------------------------
add_executable(main ${SOURCES})

# ----------------------------
# لینک به کتابخانه‌ها
# ----------------------------
target_link_libraries(main
    ${OpenCV_LIBS}
    openvino
)
>>>>>>> 4cdaf3b (CmakeList)
