add_library (saptv 
		   SHARED
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

