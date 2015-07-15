# IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
# 
# By downloading, copying, installing or using the software you agree to this license.
# If you do not agree to this license, do not download, install,
# copy or use the software.
# 
#                            License Agreement
#                 For Open Source Flow Visualization Library
# 
# Copyright 2013-2015 Abhishek Bajpayee
# 
# This file is part of openFV.
# 
# openFV is free software: you can redistribute it and/or modify it under the terms of the 
# GNU General Public License as published by the Free Software Foundation, either version 
# 3 of the License, or (at your option) any later version.
# 
# openFV is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with openFV. 
# If not, see http://www.gnu.org/licenses/.

cmake_minimum_required (VERSION 2.8)

project (sa-ptv)

set (PROJECT_VERSION_MAJOR 1)
set (PROJECT_VERSION_MINOR 0)

option(BUILD_SAMPLES "BUILD_SAMPLES" OFF)
option(BUILD_PYTHON "BUILD_PYTHON" OFF)

set(PROJECT_INCLUDE_DIR	 "${PROJECT_SOURCE_DIR}/include")

# find required packages
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
if(NOT OpenCV_FOUND)
  message("OPENCV NOT FOUND")
endif()

if(POLICY CMP0022)
  cmake_policy(SET CMP0022 OLD)
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

# prototyping executable
add_executable (sa-ptv ${PROJECT_SOURCE_DIR}/src/execs/sa-ptv.cpp)

# release files
add_executable (refocus ${PROJECT_SOURCE_DIR}/src/execs/refocus.cpp)
add_executable (jobFind ${PROJECT_SOURCE_DIR}/src/execs/jobFind.cpp)
add_executable (track ${PROJECT_SOURCE_DIR}/src/execs/track.cpp)
add_executable (calibrate ${PROJECT_SOURCE_DIR}/src/execs/calibrate.cpp)
add_executable (safeRefocus ${PROJECT_SOURCE_DIR}/src/execs/safeRefocus.cpp)

set (CERES_LIBS ceres glog gflags cholmod protobuf cxsparse gomp blas lapack)
set (OTHER_LIBS python2.7 tiff cufft cufftw ${Boost_GENERAL} ${Boost_PY})
set (PROF_LIB profiler)
set (CUSTOM_LIBS saptv cuda_saptv)
set (LIBS ${CUSTOM_LIBS} ${PROF_LIBS} ${OTHER_LIBS} ${CERES_LIBS} ${OpenCV_LIBS})

# fPIC flags, needed?
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -shared")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -shared")

# subdirectories
add_subdirectory (src)
if(BUILD_SAMPLES)
  add_subdirectory(samples)
endif()
if(BUILD_PYTHON)
  add_subdirectory(lib)
endif()

# ---------------
# Python wrappers
# ---------------

set (PYTHON_PROPS Boost_USE_STATIC_RUNTIME OFF Boost_USE_MULTITHREADING ON LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/saptv)