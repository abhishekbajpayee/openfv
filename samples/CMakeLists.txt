cmake_minimum_required (VERSION 2.8)

project (Test)

set (PROJECT_VERSION_MAJOR 1)
set (PROJECT_VERSION_MINOR 0)

find_package(OpenCV REQUIRED)

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



# include directories
include_directories( "/home/loganford16/Desktop/openfv/include" )
include_directories( ${Boost_INCLUDE_DIR} )
include_directories( ${PROJECT_INCLUDE_DIR} )
include_directories( "/home/loganford16/Desktop/openfv/bin" )
include_directories( "/usr/include/python2.7" )
include_directories( "/usr/include/opencv" )
include_directories( "/usr/include/opencv2" )
include_directories( "/usr/include/eigen3" )
include_directories( "/usr/local/cuda/include" )
include_directories( "/opt/Qt/5.4/gcc_64/include") 

set (CERES_LIBS ceres glog gflags cholmod protobuf cxsparse gomp blas lapack)
set (OTHER_LIBS python2.7 tiff cufft cufftw ${Boost_GENERAL} ${Boost_PY})
set (PROF_LIB profiler)
set (LIBS ${CUSTOM_LIBS} ${PROF_LIBS} ${OTHER_LIBS} ${CERES_LIBS} ${OpenCV_LIBS} "/home/loganford16/Desktop/openfv/bin/src/libsaptv.so")

add_executable (test ${PROJECT_SOURCE_DIR}/src/test.cpp)
target_link_libraries (test ${LIBS})
