cmake_minimum_required (VERSION 2.8)

project (sa-ptv)

set (PROJECT_VERSION_MAJOR 1)
set (PROJECT_VERSION_MINOR 0)

# find required packages
find_package (OpenCV REQUIRED)
find_package (CUDA REQUIRED)

# Required Boost Libraries
find_package(Boost)
if(Boost_FOUND)
  find_package ( Boost COMPONENTS program_options REQUIRED )
  set(Boost_PO ${Boost_LIBRARIES})
  find_package ( Boost COMPONENTS python REQUIRED )
  set(Boost_PY ${Boost_LIBRARIES})
elseif(NOT Boost_FOUND)
  message("Boost not found!")
endif()

# include directories
include_directories( ${Boost_INCLUDE_DIR} )
include_directories("/usr/include/python2.7")
include_directories ("${PROJECT_SOURCE_DIR}/include")
include_directories ("/usr/local/cuda-5.0/include")
include_directories ("/home/ab9/Apps/Qt/5.2.1/gcc_64/include")

# subdirectories
add_subdirectory (src)

find_package (OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# prototyping executable
add_executable (sa-ptv ${PROJECT_SOURCE_DIR}/src/execs/sa-ptv.cpp)

# release files
add_executable (refocus ${PROJECT_SOURCE_DIR}/src/execs/refocus.cpp)
add_executable (jobFind ${PROJECT_SOURCE_DIR}/src/execs/jobFind.cpp)
add_executable (track ${PROJECT_SOURCE_DIR}/src/execs/track.cpp)
add_executable (calibrate ${PROJECT_SOURCE_DIR}/src/execs/calibrate.cpp)

set (CERES_LIBS ceres glog gflags cholmod protobuf cxsparse gomp blas lapack)
set (OTHER_LIBS python2.7 tiff ${Boost_PO} ${Boost_PY})
set (PROF_LIB profiler)
set (CUSTOM_LIBS saptv cuda_saptv)
set (LIBS ${CUSTOM_LIBS} ${PROF_LIBS} ${OTHER_LIBS} ${CERES_LIBS} ${OpenCV_LIBS})

target_link_libraries (sa-ptv ${LIBS})
target_link_libraries (jobFind ${LIBS})
target_link_libraries (refocus ${LIBS})
target_link_libraries (track ${LIBS})
target_link_libraries (calibrate ${LIBS})

# Python wrappers

set (PYTHON_PROPS Boost_USE_STATIC_RUNTIME OFF Boost_USE_MULTITHREADING ON LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/saptv)

add_library(rendering SHARED ${PROJECT_SOURCE_DIR}/src/modules/rendering.cpp)
set_target_properties(rendering PROPERTIES ${PYTHON_PROPS} PREFIX "")
target_link_libraries(rendering ${LIBS})

add_library(refocusing SHARED ${PROJECT_SOURCE_DIR}/src/modules/refocusing.cpp)
set_target_properties(refocusing PROPERTIES ${PYTHON_PROPS} PREFIX "")
target_link_libraries(refocusing ${LIBS})