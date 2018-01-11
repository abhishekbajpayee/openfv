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

set (SRC_TO_BUILD ${PROJECT_SOURCE_DIR}/src/modules/refocusing.cpp
                  ${PROJECT_SOURCE_DIR}/src/modules/tools.cpp
                  ${PROJECT_SOURCE_DIR}/src/modules/parse_settings.cpp
                  ${PROJECT_SOURCE_DIR}/src/modules/rendering.cpp)

if (BUILD_TRACKING)
  set (SRC_TO_BUILD ${SRC_TO_BUILD} ${PROJECT_SOURCE_DIR}/src/modules/tracking.cpp)
else()
  add_definitions(-DWITHOUT_TRACKING)
endif()

if (WITH_CUDA)
  add_library (openfv SHARED ${SRC_TO_BUILD})
  target_link_libraries(openfv cuda_openfv ${LIBS})
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g -G -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52)
  cuda_add_library (cuda_openfv SHARED ${PROJECT_SOURCE_DIR}/src/modules/cuda_lib.cu)
else()
  add_library (openfv SHARED ${SRC_TO_BUILD})
  target_link_libraries(openfv ${LIBS})
endif()

set(CMAKE_INSTALL_PREFIX "/usr/local")

if(BUILD_PYTHON)
  
  execute_process(COMMAND ${PYTHON_EXEC} -c "from distutils.sysconfig import get_python_lib; print get_python_lib()" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
  message("Python package install directory: " ${PYTHON_SITE_PACKAGES})
  set(CMAKE_PYTHON_INSTALL_PREFIX ${PYTHON_SITE_PACKAGES}/openfv)

  set (PYTHON_PROPS Boost_USE_STATIC_RUNTIME OFF Boost_USE_MULTITHREADING ON LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/openfv)

  add_library(rendering SHARED ${PROJECT_SOURCE_DIR}/src/modules/rendering.cpp)
  set_target_properties(rendering PROPERTIES ${PYTHON_PROPS} PREFIX "")
  target_link_libraries(rendering ${OFV_LIBS} ${LIBS})

  add_library(refocusing SHARED ${PROJECT_SOURCE_DIR}/src/modules/refocusing.cpp)
  set_target_properties(refocusing PROPERTIES ${PYTHON_PROPS} PREFIX "")
  target_link_libraries(refocusing ${OFV_LIBS} ${LIBS})

  add_library(tools SHARED ${PROJECT_SOURCE_DIR}/src/modules/tools.cpp)
  set_target_properties(tools PROPERTIES ${PYTHON_PROPS} PREFIX "")
  target_link_libraries(tools ${OFV_LIBS} ${LIBS})

  if(BUILD_TRACKING)
    add_library(tracking SHARED ${PROJECT_SOURCE_DIR}/src/modules/tracking.cpp)
    set_target_properties(tracking PROPERTIES ${PYTHON_PROPS} PREFIX "")
    target_link_libraries(tracking ${OFV_LIBS} ${LIBS})
  endif()

  # add_library(localization SHARED ${PROJECT_SOURCE_DIR}/src/modules/tracking.cpp)
  # set_target_properties(localization PROPERTIES ${PYTHON_PROPS} PREFIX "")
  # target_link_libraries(localization ${LIBS})

  include(cmake/boost-python.cmake)

  # Build np<=>opencv converter library
  include_directories(${PROJECT_SOURCE_DIR}/include)
  boost_python_module(np_opencv_converter numpy-opencv-converter/np_opencv_converter.cpp numpy-opencv-converter/conversion.cpp)
  set_target_properties(np_opencv_converter PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/openfv)
  target_link_libraries(np_opencv_converter boost_system boost_python ${OpenCV_LDFLAGS})
  
  # Build test library
  # include_directories(${PROJECT_SOURCE_DIR}/include/numpy-opencv-converter)
  # boost_python_module(np_opencv_module tests/np_opencv_module.cpp)
  # target_link_libraries(np_opencv_module boost_system boost_python np_opencv_converter ${OpenCV_LDFLAGS})

  install(CODE "file(WRITE ${PROJECT_SOURCE_DIR}/bin/openfv/__init__.py \"import np_opencv_converter\n\")")

  if(BUILD_TRACKING)
    install(TARGETS rendering refocusing tools np_opencv_converter tracking
      DESTINATION ${CMAKE_PYTHON_INSTALL_PREFIX})
  else()
    install(TARGETS rendering refocusing tools np_opencv_converter
      DESTINATION ${CMAKE_PYTHON_INSTALL_PREFIX})
  endif()

  install(FILES ${PROJECT_SOURCE_DIR}/bin/openfv/__init__.py DESTINATION ${CMAKE_PYTHON_INSTALL_PREFIX})

endif()

install(TARGETS openfv DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

if (WITH_CUDA)
  install(TARGETS cuda_openfv DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()
