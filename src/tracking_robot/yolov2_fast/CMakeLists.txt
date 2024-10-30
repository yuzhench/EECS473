cmake_minimum_required(VERSION 3.10)

# Project name
project(YoloCppProject)

# Set C++ standard to C++11 (as per the flag -std=gnu++11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable OpenMP (equivalent to -fopenmp)
find_package(OpenMP REQUIRED)

# Compiler options (-Wall, -Wno-unknown-pragmas, -fPIE, -pthread, etc.)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas -fPIE -pthread -march=armv8-a -fno-rtti -O3 -DNDEBUG")

# Include directories for OpenCV and NCNN
include_directories(/usr/local/include/opencv4)
include_directories(/usr/local/include/ncnn)

# Link libraries
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ldl -lpthread -lgomp -rdynamic -march=armv8-a")

# Add the NCNN static library
add_library(ncnn STATIC IMPORTED)
set_target_properties(ncnn PROPERTIES IMPORTED_LOCATION /usr/local/lib/ncnn/libncnn.a)

# Find OpenCV (this replaces pkg-config --libs --cflags opencv4)
find_package(OpenCV REQUIRED)

# Add your executable
add_executable(my_program yolov5.cpp)

# Link NCNN and OpenCV libraries to the executable
target_link_libraries(my_program ncnn ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
