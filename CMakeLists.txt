cmake_minimum_required (VERSION 2.6)

project (sa-ptv)

set (PROJECT_VERSION_MAJOR 1)
set (PROJECT_VERSION_MINOR 0)

# find required packages
find_package (OpenCV REQUIRED)
find_package (CUDA REQUIRED)

# include directories
include_directories ("${PROJECT_SOURCE_DIR}/include")
include_directories ("/usr/local/cuda-5.0/include")

# subdirectories
add_subdirectory (src)

find_package (OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# prototyping executables
add_executable (sa-ptv ${PROJECT_SOURCE_DIR}/src/execs/sa-ptv.cpp)
#add_executable (testing ${PROJECT_SOURCE_DIR}/src/testing.cpp)

# release files
# add_executable (refocus ${PROJECT_SOURCE_DIR}/src/execs/refocus.cpp)
add_executable (track ${PROJECT_SOURCE_DIR}/src/execs/track.cpp)
add_executable (liveRefocus ${PROJECT_SOURCE_DIR}/src/execs/liveRefocus.cpp)
add_executable (calibrate ${PROJECT_SOURCE_DIR}/src/execs/calibrate.cpp)

set (CERES_LIBS ceres glog gflags cholmod protobuf cxsparse gomp)
set (OTHER_LIBS python2.7 tiff)
set (PROF_LIB -lprofiler)
set (CUSTOM_LIBS saptv cuda_saptv)

target_link_libraries (sa-ptv ${CUSTOM_LIBS} ${OpenCV_LIBS} ${CERES_LIBS} ${OTHER_LIBS} ${PROF_LIB})
# target_link_libraries (testing ${OpenCV_LIBS} lib ${CERES_LIBS} ${PYTHON_LIB})
# target_link_libraries (refocus ${CUSTOM_LIBS} ${OpenCV_LIBS} lib ${CERES_LIBS} ${OTHER_LIBS})
target_link_libraries (track ${CUSTOM_LIBS} ${OpenCV_LIBS} ${CERES_LIBS} ${OTHER_LIBS})
target_link_libraries (liveRefocus ${CUSTOM_LIBS} ${OpenCV_LIBS} ${CERES_LIBS} ${OTHER_LIBS})
target_link_libraries (calibrate ${CUSTOM_LIBS} ${OpenCV_LIBS} ${CERES_LIBS} ${OTHER_LIBS})