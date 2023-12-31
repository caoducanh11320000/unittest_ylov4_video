cmake_minimum_required(VERSION 2.6)

project(yolov4)

enable_language(CXX)  

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)
# Tìm gói OpenCV 1
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
#set de nang cap su chi tiet cho du an?
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
# tao thu vien shared lib
cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu ${PROJECT_SOURCE_DIR}/mish.cu)
target_link_libraries(myplugins nvinfer cudart)

# find_package(OpenCV)
# include_directories(${OpenCV_INCLUDE_DIRS})
# add_executable(yolov4 ${PROJECT_SOURCE_DIR}/yolov4.cpp)

FILE(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/include/*.h)  #thu ko  bao gom ca .h xem co duoc ko

add_executable(yolov4 ${SRC_FILES})
target_link_libraries(yolov4 nvinfer)
target_link_libraries(yolov4 cudart)
target_link_libraries(yolov4 myplugins)
target_link_libraries(yolov4 ${OpenCV_LIBS})
target_link_libraries(yolov4 stdc++fs)
