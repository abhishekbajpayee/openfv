set (CERES_LIBS ceres glog gflags cholmod protobuf cxsparse gomp blas lapack)
set (OTHER_LIBS python2.7 tiff cufft cufftw ${Boost_GENERAL} ${Boost_PY})
set (PROF_LIB profiler)
set (CUSTOM_LIBS saptv cuda_saptv)
set (LIBS ${CUSTOM_LIBS} ${PROF_LIBS} ${OTHER_LIBS} ${CERES_LIBS} ${OpenCV_LIBS})

include_directories( ${Boost_INCLUDE_DIR} )
include_directories( "/usr/include/python2.7" )
include_directories( "/usr/include/opencv" )
include_directories( ${PROJECT_INCLUDE_DIR} )
include_directories( "/usr/local/cuda/include" )
include_directories( "/home/ab9/Apps/Qt/5.2.1/gcc_64/include" )

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
