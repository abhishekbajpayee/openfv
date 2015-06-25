cmake_minimum_required (VERSION 2.8)

project (sa-ptv)

set (PROJECT_VERSION_MAJOR 1)
set (PROJECT_VERSION_MINOR 0)

# find required packages
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
if(NOT OpenCV_FOUND)
  message("OPENCV NOT FOUND")
endif()

# Required Boost Libraries
find_package(Boost)
if(Boost_FOUND)
  find_package ( Boost COMPONENTS program_options filesystem system serialization REQUIRED )
  set(Boost_GENERAL ${Boost_LIBRARIES})
  find_package ( Boost COMPONENTS python REQUIRED )
  set(Boost_PY ${Boost_LIBRARIES})
  # for some other day
  # find_package ( Boost COMPONENTS serialization REQUIRED )
  # set(Boost_SR ${Boost_LIBRARIES})
elseif(NOT Boost_FOUND)
  message("Boost not found!")
endif() 

set(PROJECT_INCLUDE_DIR	 "${PROJECT_SOURCE_DIR}/include")
set(PROJECT_SOURCE_DIR   "${PROJECT_SOURCE_DIR}/src")


# include directories
include_directories( ${Boost_INCLUDE_DIR} )
include_directories( ${PROJECT_INCLUDE_DIR} )
include_directories( "/usr/include/python2.7" )
include_directories( "/usr/include/opencv" )
include_directories( "/usr/include/opencv2" )
include_directories( "/usr/include/eigen3" )
include_directories( "/usr/local/cuda/include" )
include_directories( "/opt/Qt/5.4/gcc_64/include" )

find_package (OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS})
  set (CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
endif()

# fPIC flags, needed?
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -shared")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -shared")


# subdirectories
add_subdirectory (src)


set (CERES_LIBS ceres glog gflags cholmod protobuf cxsparse gomp blas lapack)
set (OTHER_LIBS python2.7 tiff cufft cufftw ${Boost_GENERAL} ${Boost_PY})
set (PROF_LIB profiler)
set (CUSTOM_LIBS saptv cuda_saptv)
set (LIBS ${CUSTOM_LIBS} ${PROF_LIBS} ${OTHER_LIBS} ${CERES_LIBS} ${OpenCV_LIBS})

add_library(rendering SHARED ${PROJECT_SOURCE_DIR}/modules/rendering.cpp)
set_target_properties(rendering PROPERTIES ${PYTHON_PROPS} PREFIX "")
target_link_libraries(rendering ${LIBS})

add_library(refocusing SHARED ${PROJECT_SOURCE_DIR}/modules/refocusing.cpp)
set_target_properties(refocusing PROPERTIES ${PYTHON_PROPS} PREFIX "")
target_link_libraries(refocusing ${LIBS})

add_library(tools SHARED ${PROJECT_SOURCE_DIR}/modules/tools.cpp)
set_target_properties(tools PROPERTIES ${PYTHON_PROPS} PREFIX "")
target_link_libraries(tools ${LIBS})

add_library(tracking SHARED ${PROJECT_SOURCE_DIR}/modules/tracking.cpp)
set_target_properties(tracking PROPERTIES ${PYTHON_PROPS} PREFIX "")
target_link_libraries(tracking ${LIBS})

add_library(calibration SHARED ${PROJECT_SOURCE_DIR}/modules/calibration.cpp)
set_target_properties(calibration PROPERTIES ${PYTHON_PROPS} PREFIX "")
target_link_libraries(calibration ${LIBS})

# add_library(localization SHARED ${PROJECT_SOURCE_DIR}/modules/pLoc.cpp)
# set_target_properties(localization PROPERTIES ${PYTHON_PROPS} PREFIX "")
# target_link_libraries(localization ${LIBS})

# ---------------
# Python wrappers
# ---------------

set (PYTHON_PROPS Boost_USE_STATIC_RUNTIME OFF Boost_USE_MULTITHREADING ON LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/saptv)

