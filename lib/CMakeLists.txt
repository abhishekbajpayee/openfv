add_library (saptv 
		   SHARED
		   ${PROJECT_SOURCE_DIR}/src/modules/optimization.cpp
                   ${PROJECT_SOURCE_DIR}/src/modules/calibration.cpp
		   ${PROJECT_SOURCE_DIR}/src/modules/refocusing.cpp
                   ${PROJECT_SOURCE_DIR}/src/modules/pLoc.cpp
                   ${PROJECT_SOURCE_DIR}/src/modules/tracking.cpp
                   ${PROJECT_SOURCE_DIR}/src/modules/visualize.cpp
		   ${PROJECT_SOURCE_DIR}/src/modules/tools.cpp
                   ${PROJECT_SOURCE_DIR}/src/modules/batchProc.cpp
                   ${PROJECT_SOURCE_DIR}/src/modules/parse_settings.cpp
                   ${PROJECT_SOURCE_DIR}/src/modules/rendering.cpp
                   ${PROJECT_SOURCE_DIR}/src/modules/parse_safe_settings.cpp
	  	   ${PROJECT_SOURCE_DIR}/src/modules/featureDetection.cpp
	  	   ${PROJECT_SOURCE_DIR}/src/modules/safeRefocusing.cpp
                   ${PROJECT_SOURCE_DIR}/src/modules/piv.cpp)

target_link_libraries( saptv ${LIBS})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g -G -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35)
cuda_add_library (cuda_saptv SHARED ${PROJECT_SOURCE_DIR}/src/modules/cuda_lib.cu)

if(BUILD_PYTHON)
  add_library(rendering SHARED ${PROJECT_SOURCE_DIR}/src/modules/rendering.cpp)
  set_target_properties(rendering PROPERTIES ${PYTHON_PROPS} PREFIX "")
  target_link_libraries(rendering ${LIBS})

  add_library(refocusing SHARED ${PROJECT_SOURCE_DIR}/src/modules/refocusing.cpp)
  set_target_properties(refocusing PROPERTIES ${PYTHON_PROPS} PREFIX "")
  target_link_libraries(refocusing ${LIBS})

  add_library(tools SHARED ${PROJECT_SOURCE_DIR}/src/modules/tools.cpp)
  set_target_properties(tools PROPERTIES ${PYTHON_PROPS} PREFIX "")
  target_link_libraries(tools ${LIBS})

  add_library(tracking SHARED ${PROJECT_SOURCE_DIR}/src/modules/tracking.cpp)
  set_target_properties(tracking PROPERTIES ${PYTHON_PROPS} PREFIX "")
  target_link_libraries(tracking ${LIBS})

  add_library(calibration SHARED ${PROJECT_SOURCE_DIR}/src/modules/calibration.cpp)
  set_target_properties(calibration PROPERTIES ${PYTHON_PROPS} PREFIX "")
  target_link_libraries(calibration ${LIBS})

# add_library(localization SHARED ${PROJECT_SOURCE_DIR}/src/modules/pLoc.cpp)
# set_target_properties(localization PROPERTIES ${PYTHON_PROPS} PREFIX "")
# target_link_libraries(localization ${LIBS})
endif()
