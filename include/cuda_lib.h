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
__global__ void calc_refocus_map_kernel(DevMem2Df xmap, DevMem2Df ymap, float z, int n);

__device__ point point_refrac(point Xcam, point p, float &f, float &g, float zW_, float n1_, float n2_, float n3_, float t_);

__device__ point point_refrac_fast(point Xcam, point p, float &f, float &g);

// Host wrappers
void uploadRefractiveData(float hinv[6], float locations[9][3], float pmats[9][12], float geom[5]);

void gpu_calc_refocus_map(GpuMat &xmap, GpuMat &ymap, float z, int i);

void gpu_calc_refocus_maps(vector<GpuMat> &xmaps, vector<GpuMat> &ymaps, float z);

#endif
