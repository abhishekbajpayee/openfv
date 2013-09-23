#ifndef CUDA_LIBRARY
#define CUDA_LIBRARY

#include <cuda.h>
#include <cuda_runtime.h>

#define __CUDA_INTERNAL_COMPILATION__
#include <math_functions.h>
#undef __CUDA_INTERNAL_COMPILATION__

#include <cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace cv;
using namespace gpu;

// Types
typedef struct {
    float x, y, z;
} point;

// Kernels
__global__ void warp_refractive_kernel();

__global__ void calc_refocus_map_kernel(DevMem2Df xmap, DevMem2Df ymap, DevMem2Df Hinv, DevMem2Df P, DevMem2Df PX, float z);

__device__ point point_refrac(point Xcam, point p);

// Host wrappers
void warp_refractive();

void gpu_calc_refocus_map(GpuMat &xmap, GpuMat &ymap, GpuMat &Hinv, GpuMat &P, GpuMat &PX, float z);

#endif
