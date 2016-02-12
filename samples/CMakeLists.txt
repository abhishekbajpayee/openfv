cmake_minimum_required (VERSION 2.8)

project (OpenFVSamples)

set (PROJECT_VERSION_MAJOR 1)
set (PROJECT_VERSION_MINOR 0)

find_package(OpenCV REQUIRED)

set(CUDA_INC_DIR "/usr/local/cuda/include" CACHE PATH "Path to CUDA Directory")
set(PYTHON_INC_DIR "/usr/include/python2.7" CACHE PATH "Path to Python Directory")
set(PYTHON_LIBS "/usr/lib/libpython2.7.so" CACHE PATH "Path to Python Libraries")
set(EIGEN_INC_DIR "/usr/include/eigen3" CACHE PATH "Path to Eigen Directory")
set(OFV_INC_DIR "${PROJECT_SOURCE_DIR}/../include" CACHE PATH "Path to OpenFV Include Directory")
set(OFV_LIB_DIR "${PROJECT_SOURCE_DIR}/../bin/lib" CACHE PATH "Path to OpenFV Lib Directory")

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
include_directories( ${EIGEN_INC_DIR} )

# Link directories
link_directories( ${OFV_LIB_DIR} )

set (CERES_LIBS ceres glog gflags cholmod protobuf cxsparse gomp blas lapack)
set (OTHER_LIBS tiff cufft cufftw ${Boost_GENERAL} ${Boost_PY})
set (PROF_LIB profiler)
set (LIBS ${CUSTOM_LIBS} ${PROF_LIBS} ${OTHER_LIBS} ${CERES_LIBS} ${OpenCV_LIBS} ${PYTHON_LIBS} openfv cuda_openfv)

# add_executable (tomo ${PROJECT_SOURCE_DIR}/tomotest.cpp)
# target_link_libraries (tomo ${LIBS})

add_executable (calibrate ${PROJECT_SOURCE_DIR}/calibrate.cpp)
target_link_libraries (calibrate ${LIBS})

# add_executable (track ${PROJECT_SOURCE_DIR}/track.cpp)
# target_link_libraries (track ${LIBS})

# add_executable (refocus ${PROJECT_SOURCE_DIR}/refocus.cpp)
# target_link_libraries (refocus ${LIBS})

add_executable (sa-ptv ${PROJECT_SOURCE_DIR}/sa-ptv.cpp)
target_link_libraries (sa-ptv ${LIBS})

# add_executable (jobFind ${PROJECT_SOURCE_DIR}/jobFind.cpp)
# target_link_libraries (jobFind ${LIBS})