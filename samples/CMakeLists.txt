cmake_minimum_required (VERSION 2.8)

project (OpenFVSamples)

set (PROJECT_VERSION_MAJOR 1)
set (PROJECT_VERSION_MINOR 0)

find_package(OpenCV REQUIRED)

find_package(Boost)
if(Boost_FOUND)
  find_package ( Boost COMPONENTS program_options filesystem system serialization REQUIRED )
  set(Boost_GENERAL ${Boost_LIBRARIES})
  find_package ( Boost COMPONENTS python REQUIRED )
  set(Boost_PY ${Boost_LIBRARIES})
elseif(NOT Boost_FOUND)
  message("Boost not found!")
endif() 

# Include directories
include_directories( ${OFV_INC_DIR} )
include_directories( ${PYTHON_INC_DIR} )
include_directories( ${CUDA_INC_DIR} )
include_directories( ${Boost_INCLUDE_DIR} )

# Link directories
link_directories( ${OFV_LIB_DIR} )

set (CERES_LIBS ceres glog gflags cholmod protobuf cxsparse gomp blas lapack)
set (OTHER_LIBS python2.7 tiff cufft cufftw ${Boost_GENERAL} ${Boost_PY})
set (PROF_LIB profiler)
set (LIBS ${CUSTOM_LIBS} ${PROF_LIBS} ${OTHER_LIBS} ${CERES_LIBS} ${OpenCV_LIBS} saptv)

add_executable (calibrate ${PROJECT_SOURCE_DIR}/calibrate.cpp)
target_link_libraries (calibrate ${LIBS})

add_executable (track ${PROJECT_SOURCE_DIR}/track.cpp)
target_link_libraries (track ${LIBS})

add_executable (refocus ${PROJECT_SOURCE_DIR}/refocus.cpp)
target_link_libraries (refocus ${LIBS})

add_executable (sa-ptv ${PROJECT_SOURCE_DIR}/sa-ptv.cpp)
target_link_libraries (sa-ptv ${LIBS})

add_executable (jobFind ${PROJECT_SOURCE_DIR}/jobFind.cpp)
target_link_libraries (jobFind ${LIBS})