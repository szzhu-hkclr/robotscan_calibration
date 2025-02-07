cmake_minimum_required(VERSION 3.5)
project(hand_eye_calibration)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}
    src/handeye_6DoF.cpp
    src/RobotSerial.cpp
    src/calibration_utils.cpp
    src/json_utils.cpp
    src/image_utils.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)