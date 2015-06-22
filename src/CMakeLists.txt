add_library (saptv 
		   modules/optimization.cpp
                   modules/calibration.cpp
		   modules/refocusing.cpp
                   modules/pLoc.cpp
                   modules/tracking.cpp
                   modules/visualize.cpp
		   modules/tools.cpp
                   modules/batchProc.cpp
                   modules/parse_settings.cpp
                   modules/rendering.cpp
                   modules/parse_safe_settings.cpp
	  	   modules/featureDetection.cpp
	  	   modules/safeRefocusing.cpp
                   modules/piv.cpp)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g -G -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35)
cuda_add_library (cuda_saptv modules/cuda_lib.cu)

include_directories( ${Boost_INCLUDE_DIR} )
include_directories( "/usr/include/python2.7" )
include_directories( "/usr/include/opencv" )
include_directories( ${PROJECT_INCLUDE_DIR} )
include_directories( "/usr/local/cuda/include" )
include_directories( "/home/ab9/Apps/Qt/5.2.1/gcc_64/include" )


# prototyping executable
add_executable (sa-ptv ${PROJECT_SOURCE_DIR}/execs/sa-ptv.cpp)

# release files
add_executable (refocus ${PROJECT_SOURCE_DIR}/execs/refocus.cpp)
add_executable (jobFind ${PROJECT_SOURCE_DIR}/execs/jobFind.cpp)
add_executable (track ${PROJECT_SOURCE_DIR}/execs/track.cpp)
add_executable (calibrate ${PROJECT_SOURCE_DIR}/execs/calibrate.cpp)
add_executable (safeRefocus ${PROJECT_SOURCE_DIR}/execs/safeRefocus.cpp)

set (CERES_LIBS ceres glog gflags cholmod protobuf cxsparse gomp blas lapack)
set (OTHER_LIBS python2.7 tiff cufft cufftw ${Boost_GENERAL} ${Boost_PY})
set (PROF_LIB profiler)
set (CUSTOM_LIBS saptv cuda_saptv)
set (LIBS ${CUSTOM_LIBS} ${PROF_LIBS} ${OTHER_LIBS} ${CERES_LIBS} ${OpenCV_LIBS})

target_link_libraries (sa-ptv ${LIBS})
target_link_libraries (jobFind ${LIBS})
target_link_libraries (refocus ${LIBS})
target_link_libraries (track ${LIBS})
target_link_libraries (calibrate ${LIBS})
target_link_libraries (safeRefocus ${LIBS})
