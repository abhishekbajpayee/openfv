cmake_minimum_required (VERSION 2.6)

project (sa-ptv)

set (PROJECT_VERSION_MAJOR 1)
set (PROJECT_VERSION_MINOR 0)

find_package (OpenCV REQUIRED)

# include directories
include_directories ("${PROJECT_SOURCE_DIR}/lib")
include_directories ("${PROJECT_SOURCE_DIR}/include")

# subdirectories
add_subdirectory (lib)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable (sa-ptv sa-ptv.cpp)
add_executable (refocus refocus.cpp)

set (CERES_LIBS ceres glog gflags cholmod protobuf cxsparse gomp)

set (PYTHON_LIB python2.7)

target_link_libraries (sa-ptv ${OpenCV_LIBS} lib ${CERES_LIBS} ${PYTHON_LIB})

target_link_libraries (refocus ${OpenCV_LIBS} lib ${CERES_LIBS} ${PYTHON_LIB})
