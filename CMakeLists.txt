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

project (openfv)

set (PROJECT_VERSION_MAJOR 0)
set (PROJECT_VERSION_MINOR 1)

# Build Directories
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Options
option(BUILD_PYTHON "BUILD_PYTHON" OFF)
option(FAST_MATH "FAST_MATH" OFF)
option(WITH_CUDA "WITH_CUDA" ON)

set(PYTHON_INCLUDE_DIR "/usr/include/python2.7" CACHE PATH "Path to Python Directory")
set(PYTHON_LIBRARIES "/usr/lib/libpython2.7.so" CACHE PATH "Path to Python Libraries")
set(BOOST_PYTHON_VERSION "python" CACHE PATH "Boost Python Version [python or python3]")
set(EIGEN_INC_DIR "/usr/include/eigen3" CACHE PATH "Path to Eigen Directory")
set(PROJECT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include" CACHE PATH "Project Include Directory")

if(POLICY CMP0022)
  cmake_policy(SET CMP0022 OLD)
endif()

# FIND REQUIRED PACKAGES

# CUDA
find_package(CUDA)
message("---CUDA---")
message("CUDA Libraries: " ${CUDA_LIBRARIES})
message("CUDA Include Dirs: " ${CUDA_INCLUDE_DIRS})
if(NOT CUDA_FOUND)
  message("CUDA not found! CUDA components will not be built.")
  add_definitions(-DWITHOUT_CUDA)
  set(WITH_CUDA OFF)
endif()

if (NOT WITH_CUDA)
  message("WITH_CUDA set of OFF. CUDA components will not be built.")
  add_definitions(-DWITHOUT_CUDA)
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)
message("---OpenCV---")
message("OpenCV Libraries: " ${OpenCV_LIBS})
message("OpenCV Include Dirs: " ${OpenCV_INCLUDE_DIRS})
if(NOT OpenCV_FOUND)
  message("OpenCV not found!")
endif()

# Boost Libraries
find_package(Boost)
if(Boost_FOUND)
  find_package ( Boost COMPONENTS program_options filesystem system serialization REQUIRED )
  set(Boost_GENERAL ${Boost_LIBRARIES})
  find_package ( Boost COMPONENTS ${BOOST_PYTHON_VERSION} REQUIRED )
  set(Boost_PY ${Boost_LIBRARIES})
elseif(NOT Boost_FOUND)
  message("Boost not found!")
endif()

find_package(TIFF REQUIRED)
if(NOT TIFF_FOUND)
  message("libtiff not found!")
endif()

# Include directories
include_directories( ${PROJECT_INCLUDE_DIR} )
include_directories( ${Boost_INCLUDE_DIR} )
include_directories( ${PYTHON_INCLUDE_DIR} )
include_directories( ${EIGEN_INC_DIR} )

# Don't think these are needed because OpenFV
# doesn't link to Qt
# include_directories( "/opt/Qt/5.4/gcc_64/include" )

# Omitting for now because nothing in OpenFV as
# of now uses OpenMP
# find_package (OpenMP)
# if (OPENMP_FOUND)
#   set (CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS})
#   set (CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
# endif()

set (CERES_LIBS ceres glog gflags cholmod cxsparse blas lapack gomp) # protobuf lgomp (removed)
set (OTHER_LIBS ${TIFF_LIBRARIES} ${Boost_GENERAL} ${Boost_PY})
if (WITH_CUDA)
  set(OTHER_LIBS ${OTHER_LIBS} cufft cufftw)
  set (OFV_LIBS openfv cuda_openfv)
else()
  set (OFV_LIBS openfv)
endif()
set (PROF_LIB profiler)
set (LIBS ${PROF_LIBS} ${OTHER_LIBS} ${PYTHON_LIBRARIES} ${CERES_LIBS} ${OpenCV_LIBS})

# fPIC flags, needed?
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

if (FAST_MATH)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
endif()

set(CMAKE_INSTALL_PREFIX "/usr/local/openfv")

# Subdirectories
add_subdirectory(src)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
