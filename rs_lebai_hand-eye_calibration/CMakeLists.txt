cmake_minimum_required(VERSION 3.0.2)
project(lebai_caibration)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CAMKE_CXX_FLAGS} -std=c++11 -pthread")
find_package(OpenCV REQUIRED PATHS "/home/ma/1Citrus picking robot/opencv4.2.0/opencv-4.2.0/build" NO_DEFAULT_PATH)
find_package(realsense2 REQUIRED)
find_package(Threads REQUIRED)
include_directories(
include
  ${catkin_INCLUDE_DIRS}
  {OpenCV_INCLUDE_DIRS}
 /usr/lib/cmake/lebai/
  /usr/include/eigen3
)
add_library(${PROJECT_NAME} SHARED IMPORTED
)
add_executable(ku src/main.cpp src/CoordinatesTransform.cpp src/getB2C.cpp src/getC2F.cpp src/Camera.cpp src/CameraCalibration.cpp)
# add_executable(ku src/main.cpp src/CoordinatesTransform.cpp src/getB2C.cpp src/getC2F.cpp src/Camera.cpp )
set_target_properties(${PROJECT_NAME} PROPERTIES IMPORTED_LOCATION /usr/lib/liblebai-cpp.so )
target_link_libraries(ku
  ${PROJECT_NAME}
   ${OpenCV_LIBS}
   ${realsense2_LIBRARY}
)


