cmake_minimum_required(VERSION 3.0)
include(CMakePrintHelpers)

project(OnnxTest1 CXX)
set (version 1.0)

set(CMAKE_CXX_STANDARD 17)

find_package(onnxruntime 1.0 REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "The include folder is ${onnxinclude}")
cmake_print_variables(OpenCV_INCLUDE_DIRS)

add_executable(onnxtest1 onnxtest1.cpp)
target_include_directories(onnxtest1 PRIVATE ${onnxinclude})
target_link_libraries(onnxtest1 PRIVATE onnxlib)

add_executable(onnxtest2 onnxtest2.cpp)
target_include_directories(onnxtest2 PRIVATE ${onnxinclude})
target_link_libraries(onnxtest2 PRIVATE onnxlib)

add_executable(onnxtest3 onnxtest3.cpp)
target_include_directories(onnxtest3 PRIVATE ${OpenCV_INCLUDE_DIRS} ${onnxinclude})
target_link_libraries(onnxtest3 PRIVATE ${OpenCV_LIBS} onnxlib)

add_executable(onnxtest4 onnxtest4.cpp)
target_include_directories(onnxtest4 PRIVATE ${OpenCV_INCLUDE_DIRS} ${onnxinclude})
target_link_libraries(onnxtest4 PRIVATE ${OpenCV_LIBS} onnxlib)

add_executable(onnxperformance onnxperformance.cpp)
target_include_directories(onnxperformance PRIVATE ${OpenCV_INCLUDE_DIRS} ${onnxinclude})
target_link_libraries(onnxperformance PRIVATE ${OpenCV_LIBS} onnxlib)

add_executable(onnxperformance_yolov8 onnxperformance_yolov8.cpp)
target_include_directories(onnxperformance_yolov8 PRIVATE ${OpenCV_INCLUDE_DIRS} ${onnxinclude})
target_link_libraries(onnxperformance_yolov8 PRIVATE ${OpenCV_LIBS} onnxlib)

add_executable(inference_yolov8 InferenceYolov8.cpp)
target_include_directories(inference_yolov8 PRIVATE ${OpenCV_INCLUDE_DIRS} ${onnxinclude})
target_link_libraries(inference_yolov8 PRIVATE ${OpenCV_LIBS} onnxlib)
