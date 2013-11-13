find_package (OpenCV REQUIRED)
find_package (CUDA REQUIRED)

add_library (saptv modules/optimization.cpp
                   modules/calibration.cpp
		   modules/refocusing.cpp
                   modules/pLoc.cpp
                   modules/tracking.cpp
                   modules/visualize.cpp
		   modules/tools.cpp)

#add_library (saptv_shared SHARED modules/py_sa-ptv.cpp)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g -G -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35)
cuda_add_library (cuda_saptv modules/cuda_lib.cu)

target_link_libraries (saptv ${OpenCV_LIBS})