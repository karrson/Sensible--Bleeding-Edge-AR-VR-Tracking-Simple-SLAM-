cmake_minimum_required(VERSION 3.4)
project(BST_SLAM_2D CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(${PROJECT_NAME} Main.cpp)
find_package(OpenCV CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE opencv_core)

# === Sensor Driver Packages ===
find_package(depthai CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE depthai::opencv)
