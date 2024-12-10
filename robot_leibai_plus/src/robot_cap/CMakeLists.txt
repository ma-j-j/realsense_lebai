cmake_minimum_required(VERSION 3.0.2)
project(robot_cap)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE "Release")

find_package(catkin REQUIRED COMPONENTS
        geometry_msgs
        message_generation
        roscpp
        rospy
        std_msgs
        sensor_msgs
        cv_bridge
        image_transport
)
# opencv
set(OpenCV_DIR /home/ma/1Citrus picking robot/opencv4.2.0/opencv-4.2.0/build)
find_package(OpenCV REQUIRED PATHS "/home/ma/1Citrus picking robot/opencv4.2.0/opencv-4.2.0/build" NO_DEFAULT_PATH)
include_directories(${OpenCV_INCLUDE_DIRS})

# realsense
find_package(realsense2 REQUIRED)
find_package(Threads REQUIRED)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

# tensorrt
include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)
include_directories(/home/ma/TensorRT-8.2.1.8/include/)

# pcl
find_package(PCL 1.13 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})





include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

# add_message_files(
#   FILES
#   objection.msg
#   objectionsofonemat.msg
# )
# generate_messages(
#   DEPENDENCIES
#   std_msgs
#   geometry_msgs
#   sensor_msgs
# )
catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES realsense_d435
  CATKIN_DEPENDS geometry_msgs message_generation message_runtime roscpp rospy std_msgs sensor_msgs cv_bridge image_transport
#  DEPENDS system_lib
)


# tensorrt_link
include_directories(${PROJECT_SOURCE_DIR}/trt-src/)
include_directories(${PROJECT_SOURCE_DIR}/trt-plugin/)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/trt-src/*.cpp ${PROJECT_SOURCE_DIR}/trt-src/*.cu)
file(GLOB_RECURSE PLUGIN_SRCS ${PROJECT_SOURCE_DIR}/trt-plugin/*.cu)
add_library(myplugins SHARED ${PLUGIN_SRCS} ${SRCS} )
target_link_libraries(myplugins nvinfer cudart)

#lebai SDK
include_directories(
include ${catkin_INCLUDE_DIRS}
 /usr/lib/cmake/lebai/
  /usr/include/eigen3
)
add_library(${PROJECT_NAME} SHARED IMPORTED
)
set_target_properties(${PROJECT_NAME} PROPERTIES IMPORTED_LOCATION /usr/lib/liblebai-cpp.so )

add_executable(citrus main.cpp  ${myplugins} )
target_link_libraries(citrus nvinfer)
target_link_libraries(citrus cudart)
target_link_libraries(citrus myplugins)
target_link_libraries(citrus ${OpenCV_LIBS} ${PCL_LIBRARIES} ${realsense2_LIBRARY} ${PROJECT_NAME} ${catkin_LIBRARIES})
add_executable(sub sub.cpp  ${myplugins} )
target_link_libraries(sub ${OpenCV_LIBS} ${PCL_LIBRARIES} ${realsense2_LIBRARY} ${PROJECT_NAME} ${catkin_LIBRARIES})
add_executable(obb obb.cpp  ${myplugins} )
target_link_libraries(obb ${OpenCV_LIBS} ${PCL_LIBRARIES} ${realsense2_LIBRARY} ${PROJECT_NAME} ${catkin_LIBRARIES})


## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/robot_cap.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/robot_cap_node.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

