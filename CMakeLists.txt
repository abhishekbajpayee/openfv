#  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
#
#  By downloading, copying, installing or using the software you agree to this license.
#  If you do not agree to this license, do not download, install,
#  copy or use the software.
#
#                           License Agreement
#                For Open Source Flow Visualization Library
#
# Copyright 2013-2018 Abhishek Bajpayee
#
# This file is part of OpenFV.
#
# OpenFV is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License version 2 as published by the Free Software Foundation.
#
# OpenFV is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License version 2 for more details.
#
# You should have received a copy of the GNU General Public License version 2 along with
# OpenFV. If not, see https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html.

cmake_minimum_required (VERSION 2.8)

project (openfv)

set (PROJECT_VERSION_MAJOR 0)
set (PROJECT_VERSION_MINOR 1)

# Build Directories
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Options
option(BUILD_PYTHON "BUILD_PYTHON" ON)
option(BUILD_TRACKING "BUILD_TRACKING" OFF)
option(WITH_CUDA "WITH_CUDA" ON)

set(EIGEN_INC_DIR "/usr/include/eigen3" CACHE PATH "Path to Eigen Directory")
set(PYTHON_EXEC "python2.7" CACHE PATH "Python executable (used to find which dir to install python bindings in)")
set(PROJECT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include/openfv" CACHE PATH "Project Include Directory")

if(POLICY CMP0022)
  cmake_policy(SET CMP0022 OLD)
endif()

# FIND REQUIRED PACKAGES

# Python
if (BUILD_PYTHON)
  set(Python_ADDITIONAL_VERSIONS "2.7")
  find_package(PythonLibs REQUIRED)
  message("---PYTHON---")
  message("Python Libraries: " ${PYTHON_LIBRARIES})
  message("Python Include Dirs: " ${PYTHON_INCLUDE_DIRS})
  if (NOT PYTHONLIBS_FOUND)
    message("Python not found! Python components will not be built.")
    set(BUILD_PYTHON OFF)
  elseif(PYTHONLIBS_FOUND)
    add_definitions(-DWITH_PYTHON)
  endif()
  set(BOOST_PYTHON_VERSION "python" CACHE PATH "Boost Python Version [python or python3]")
endif()

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
  find_package(Boost COMPONENTS program_options filesystem system serialization chrono REQUIRED)
  set(Boost_GENERAL ${Boost_LIBRARIES})
  if(BUILD_PYTHON)
    find_package(Boost COMPONENTS ${BOOST_PYTHON_VERSION} REQUIRED)
    set(Boost_PY ${Boost_LIBRARIES})
  endif()
elseif(NOT Boost_FOUND)
  message("Boost not found!")
endif()

# Tiff Library
find_package(TIFF REQUIRED)
if(NOT TIFF_FOUND)
  message("libtiff not found!")
endif()

if(BUILD_TRACKING)
  message("---Ceres Solver---")
  find_package(Ceres REQUIRED)
  if(Ceres_FOUND)
    message("Ceres Libraries: " ${CERES_LIBRARIES})
    message("Ceres Include Dirs: " ${CERES_INCLUDES})
  else()
    message("Ceres Solver not found!")
  endif()
endif()

# Include directories
include_directories( ${PROJECT_INCLUDE_DIR} )
include_directories( ${Boost_INCLUDE_DIR} )
include_directories( ${PYTHON_INCLUDE_DIRS} )
include_directories( ${EIGEN_INC_DIR} )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${CUDA_INCLUDE_DIRS} )
include_directories( ${CERES_INCLUDES} )

# Don't think these are needed because OpenFV
# doesn't link to Qt
# include_directories( "/opt/Qt/5.4/gcc_64/include" )

set(OTHER_LIBS ${TIFF_LIBRARIES} ${Boost_GENERAL} ${Boost_PY} yaml-cpp glog gflags)
if(WITH_CUDA)
  set(OTHER_LIBS ${OTHER_LIBS})
  set (OFV_LIBS openfv cuda_openfv)
else()
  set (OFV_LIBS openfv)
endif()
if(BUILD_PYTHON)
  set (OTHER_LIBS ${OTHER_LIBS} ${PYTHON_LIBRARIES})
endif()
set(PROF_LIB profiler)
if(BUILD_TRACKING)
  set(CERES_LIBS ${CERES_LIBRARIES} cholmod cxsparse blas lapack gomp)
  set(LIBS ${PROF_LIBS} ${OTHER_LIBS} ${CERES_LIBS} ${OpenCV_LIBS})
else()
  set(LIBS ${PROF_LIBS} ${OTHER_LIBS} ${OpenCV_LIBS})
endif()

# fPIC flags, needed?
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# Subdirectories
add_subdirectory(src)

add_executable(sa_reconstruct ${PROJECT_SOURCE_DIR}/src/tools/sa_reconstruct.cpp)
target_link_libraries(sa_reconstruct ${LIBS} ${OFV_LIBS})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/openfv/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include/openfv)
