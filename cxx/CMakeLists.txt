cmake_minimum_required(VERSION 3.5)
project(roboscan_calib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_subdirectory(cnpy)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/cnpy
)

add_executable(${PROJECT_NAME}
    src/roboscan_calib.cpp
    src/handeye_newkin.cpp
    src/handeye_6DoF.cpp
    src/RobotSerial.cpp
    src/calibration_utils.cpp
    src/json_utils.cpp
    src/image_utils.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    cnpy
)
