//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//                           License Agreement
//                For Open Source Flow Visualization Library
//
// Copyright 2013-2017 Abhishek Bajpayee
//
// This file is part of OpenFV.
//
// OpenFV is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License version 2 as published by the Free Software Foundation.
//
// OpenFV is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License version 2 for more details.
//
// You should have received a copy of the GNU General Public License version 2 along with
// OpenFV. If not, see https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html.

#ifndef CUDA_LIBRARY
#define CUDA_LIBRARY

#include <cuda.h>
#include <cuda_runtime.h>

// #define __CUDA_INTERNAL_COMPILATION__
// #include <math_functions.h>
// #undef __CUDA_INTERNAL_COMPILATION__

#include <cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <iostream>

using namespace cv;
using namespace gpu;

// Types
typedef struct {
    float x, y, z;
} point;

// Kernels
__global__ void calc_refocus_map_kernel(PtrStepSzf xmap, PtrStepSzf ymap, float z, int n, int rows, int cols);

__device__ point point_refrac(point Xcam, point p, float &f, float &g, float zW_, float n1_, float n2_, float n3_, float t_);

__device__ point point_refrac_fast(point Xcam, point p, float &f, float &g);

__global__ void calc_nlca_image_fast(PtrStepSzf nlca_image, PtrStepSzf img1, PtrStepSzf img2, PtrStepSzf img3, PtrStepSzf img4, int rows, int cols, float sigma);

__global__ void calc_nlca_image(PtrStepSzf nlca_image, PtrStepSzf img1, PtrStepSzf img2, PtrStepSzf img3, PtrStepSzf img4, int rows, int cols, int window, float sigma);

// Host wrappers
void uploadRefractiveData(float hinv[6], float locations[9][3], float pmats[9][12], float geom[5]);

void gpu_calc_refocus_map(GpuMat &xmap, GpuMat &ymap, float z, int i, int rows, int cols);

void gpu_calc_refocus_maps(vector<GpuMat> &xmaps, vector<GpuMat> &ymaps, float z);

void gpu_calc_nlca_image_fast(vector<GpuMat> &warped, GpuMat &nlca_image, int rows, int cols, float sigma);

void gpu_calc_nlca_image(vector<GpuMat> &warped, GpuMat &nlca_image, int rows, int cols, int window, float sigma);

#endif
