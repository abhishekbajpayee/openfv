//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//                           License Agreement
//                For Open Source Flow Visualization Library
//
// Copyright 2013-2015 Abhishek Bajpayee
//
// This file is part of openFV.
//
// openFV is free software: you can redistribute it and/or modify it under the terms of the 
// GNU General Public License as published by the Free Software Foundation, either version 
// 3 of the License, or (at your option) any later version.
//
// openFV is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
// See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with openFV. 
// If not, see http://www.gnu.org/licenses/.

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
__global__ void calc_refocus_map_kernel(PtrStepSzf xmap, PtrStepSzf ymap, float z, int n);

__device__ point point_refrac(point Xcam, point p, float &f, float &g, float zW_, float n1_, float n2_, float n3_, float t_);

__device__ point point_refrac_fast(point Xcam, point p, float &f, float &g);

// Host wrappers
void uploadRefractiveData(float hinv[6], float locations[9][3], float pmats[9][12], float geom[5]);

void gpu_calc_refocus_map(GpuMat &xmap, GpuMat &ymap, float z, int i);

void gpu_calc_refocus_maps(vector<GpuMat> &xmaps, vector<GpuMat> &ymaps, float z);

#endif
