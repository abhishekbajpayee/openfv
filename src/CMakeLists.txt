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

if (WITH_CUDA)
  add_library (openfv SHARED
    ${PROJECT_SOURCE_DIR}/src/modules/optimization.cpp
    ${PROJECT_SOURCE_DIR}/src/modules/calibration.cpp
    ${PROJECT_SOURCE_DIR}/src/modules/refocusing.cpp
    ${PROJECT_SOURCE_DIR}/src/modules/pLoc.cpp
    ${PROJECT_SOURCE_DIR}/src/modules/tracking.cpp
    ${PROJECT_SOURCE_DIR}/src/modules/visualize.cpp
    ${PROJECT_SOURCE_DIR}/src/modules/tools.cpp
    # ${PROJECT_SOURCE_DIR}/src/modules/batchProc.cpp
    ${PROJECT_SOURCE_DIR}/src/modules/parse_settings.cpp
    ${PROJECT_SOURCE_DIR}/src/modules/rendering.cpp
    ${PROJECT_SOURCE_DIR}/src/modules/parse_safe_settings.cpp
    ${PROJECT_SOURCE_DIR}/src/modules/featureDetection.cpp
    ${PROJECT_SOURCE_DIR}/src/modules/safeRefocusing.cpp)
    # ${PROJECT_SOURCE_DIR}/src/modules/piv.cpp)
  target_link_libraries(openfv cuda_openfv ${LIBS})
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g -G -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52)
  cuda_add_library (cuda_openfv SHARED ${PROJECT_SOURCE_DIR}/src/modules/cuda_lib.cu)
else()
  add_library (openfv SHARED
    ${PROJECT_SOURCE_DIR}/src/modules/optimization.cpp
    ${PROJECT_SOURCE_DIR}/src/modules/calibration.cpp
    ${PROJECT_SOURCE_DIR}/src/modules/refocusing.cpp
    ${PROJECT_SOURCE_DIR}/src/modules/pLoc.cpp
    ${PROJECT_SOURCE_DIR}/src/modules/tracking.cpp
    ${PROJECT_SOURCE_DIR}/src/modules/visualize.cpp
    ${PROJECT_SOURCE_DIR}/src/modules/tools.cpp
    # ${PROJECT_SOURCE_DIR}/src/modules/batchProc.cpp
    ${PROJECT_SOURCE_DIR}/src/modules/parse_settings.cpp
    ${PROJECT_SOURCE_DIR}/src/modules/rendering.cpp
    ${PROJECT_SOURCE_DIR}/src/modules/parse_safe_settings.cpp
    ${PROJECT_SOURCE_DIR}/src/modules/featureDetection.cpp
    ${PROJECT_SOURCE_DIR}/src/modules/safeRefocusing.cpp)
  target_link_libraries(openfv ${LIBS})
endif()

if(BUILD_PYTHON)
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

  add_library(tracking SHARED ${PROJECT_SOURCE_DIR}/src/modules/tracking.cpp)
  set_target_properties(tracking PROPERTIES ${PYTHON_PROPS} PREFIX "")
  target_link_libraries(tracking ${OFV_LIBS} ${LIBS})

  add_library(calibration SHARED ${PROJECT_SOURCE_DIR}/src/modules/calibration.cpp)
  set_target_properties(calibration PROPERTIES ${PYTHON_PROPS} PREFIX "")
  target_link_libraries(calibration ${OFV_LIBS} ${LIBS})

  # add_library(localization SHARED ${PROJECT_SOURCE_DIR}/src/modules/pLoc.cpp)
  # set_target_properties(localization PROPERTIES ${PYTHON_PROPS} PREFIX "")
  # target_link_libraries(localization ${LIBS})
  
  install(CODE "file(WRITE ${PROJECT_SOURCE_DIR}/bin/openfv/__init__.py \"\")")

  install(TARGETS rendering refocusing tools tracking calibration
          DESTINATION ${CMAKE_PYTHON_INSTALL_PREFIX}) 
  install(FILES ${PROJECT_SOURCE_DIR}/bin/openfv/__init__.py DESTINATION ${CMAKE_PYTHON_INSTALL_PREFIX})
endif()

install(TARGETS openfv DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

if (WITH_CUDA)
  install(TARGETS cuda_openfv DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()
