find_package (OpenCV REQUIRED)
find_package (CUDA REQUIRED)

add_library (saptv optimization.cpp
                   calibration.cpp
		   refocusing.cpp
                   pLoc.cpp
                   tracking.cpp
                   visualize.cpp
		   tools.cpp)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g -G -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35)
cuda_add_library (cuda_saptv cuda_lib.cu)

target_link_libraries (saptv ${OpenCV_LIBS})