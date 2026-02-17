cmake_minimum_required(VERSION 3.10)

# Set the project name
project(TRUCO_tests)
find_package(OpenMP REQUIRED)
# 3. Link OpenMP to your target

# --- RELEASE MODE CONFIGURATION ---
# If no build type is specified, default to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()


# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#DEV option
option(BUILD_DEV "Build the dev executables" OFF)

IF(BUILD_DEV)
add_subdirectory(dev)
ENDIF()
# 1. Find the OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# 2. Add the executables
add_executable(test_trucontour test_trucontour.cpp findtrucontour.h    bordertrackeromp.h bordertrackeromp.cpp)
# Modern CMake: use target_link_libraries which handles include paths automatically
target_link_libraries(test_trucontour PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
