// -------------------------------------------------------
// -------------------------------------------------------
// Synthetic Aperture - Particle Tracking Velocimetry Code
// --- Custom CUDA Modules ---
// -------------------------------------------------------
// Author: Abhishek Bajpayee
//         Dept. of Mechanical Engineering
//         Massachusetts Institute of Technology
// -------------------------------------------------------
// -------------------------------------------------------

#include <cuda.h>
#include <cuda_runtime.h>

#define __CUDA_INTERNAL_COMPILATION__
#include <math_functions.h>
#undef __CUDA_INTERNAL_COMPILATION__

#include <cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "cuda_lib.h"

using namespace cv;
using namespace gpu;

__device__ float zW_;
__device__ float n1_;
__device__ float n2_;
__device__ float n3_;
__device__ float t_;

__global__ void warp_refractive_kernel() {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

}

__global__ void calc_refocus_map_kernel(DevMem2Df xmap, DevMem2Df ymap, DevMem2Df Hinv, DevMem2Df P, DevMem2Df PX, float z) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    point p;
    p.x = Hinv.ptr(0)[0]*j + Hinv.ptr(0)[1]*i + Hinv.ptr(0)[2];
    p.y = Hinv.ptr(1)[0]*j + Hinv.ptr(1)[1]*i + Hinv.ptr(1)[2];
    p.z = z;

    point Xcam;
    Xcam.x = PX.ptr(0)[0];
    Xcam.y = PX.ptr(1)[0];
    Xcam.z = PX.ptr(2)[0];

    point a = point_refrac(Xcam, p);
    __syncthreads();

    printf("%f\t%f\t%f\n", a.x, a.y, a.z);

}

__device__ point point_refrac(point Xcam, point p) {

    float c[3];
    c[0] = Xcam.x; c[1] = Xcam.y; c[2] = Xcam.z;

    float a[3];
    float b[3];
    float pt[3];
    pt[0] = p.x; pt[1] = p.y; pt[2] = p.z;

    a[0] = c[0] + (pt[0]-c[0])*(zW_-c[2])/(pt[2]-c[2]);
    a[1] = c[1] + (pt[1]-c[1])*(zW_-c[2])/(pt[2]-c[2]);
    a[2] = zW_;
    b[0] = c[0] + (pt[0]-c[0])*(t_+zW_-c[2])/(pt[2]-c[2]);
    b[1] = c[1] + (pt[1]-c[1])*(t_+zW_-c[2])/(pt[2]-c[2]);
    b[2] = t_+zW_;
        
    float rp = sqrt( powf(pt[0]-c[0],2) + powf(pt[1]-c[1],2) );
    float dp = pt[2]-b[2];
    float phi = atan2(pt[1]-c[1],pt[0]-c[0]);
    
    float ra = sqrt( powf(a[0]-c[0],2) + powf(a[1]-c[1],2) );
    float rb = sqrt( powf(b[0]-c[0],2) + powf(b[1]-c[1],2) );
    float da = a[2]-c[2];
    float db = b[2]-a[2];
    
    float f, g, dfdra, dfdrb, dgdra, dgdrb;
        
    // Newton Raphson loop to solve for Snell's law
    float tol=1E-8;
    do {
        
        f = ( ra/sqrt(powf(ra,2)+powf(da,2)) ) - ( (n2_/n1_)*(rb-ra)/sqrt(powf(rb-ra,2)+powf(db,2)) );
        g = ( (rb-ra)/sqrt(powf(rb-ra,2)+powf(db,2)) ) - ( (n3_/n2_)*(rp-rb)/sqrt(powf(rp-rb,2)+powf(dp,2)) );
        
        dfdra = ( 1.0/sqrt(pow(ra,2)+pow(da,2)) )
            - ( powf(ra,2)/powf(powf(ra,2)+powf(da,2),1.5) )
            + ( (n2_/n1_)/sqrt(powf(ra-rb,2)+powf(db,2)) )
            - ( (n2_/n1_)*(ra-rb)*(2*ra-2*rb)/(2*powf(powf(ra-rb,2)+powf(db,2),1.5)) );
        
        dfdrb = ( (n2_/n1_)*(ra-rb)*(2*ra-2*rb)/(2*powf(powf(ra-rb,2)+powf(db,2),1.5)) )
            - ( (n2_/n1_)/sqrt(powf(ra-rb,2)+powf(db,2)) );
        
        dgdra = ( (ra-rb)*(2*ra-2*rb)/(2*powf(powf(ra-rb,2.0)+powf(db,2.0),1.5)) )
            - ( (1.0)/sqrt(powf(ra-rb,2.0)+powf(db,2.0)) );
        
        dgdrb = ( (1.0)/sqrt(powf(ra-rb,2)+powf(db,2)) )
            + ( (n3_/n2_)/sqrt(powf(rb-rp,2)+powf(dp,2)) )
            - ( (ra-rb)*(2*ra-2*rb)/(2*powf(powf(ra-rb,2)+powf(db,2),1.5)) )
            - ( (n3_/n2_)*(rb-rp)*(2*rb-2*rp)/(2*powf(powf(rb-rp,2)+powf(dp,2),1.5)) );
        
        ra = ra - ( (f*dgdrb - g*dfdrb)/(dfdra*dgdrb - dfdrb*dgdra) );
        rb = rb - ( (g*dfdra - f*dgdra)/(dfdra*dgdrb - dfdrb*dgdra) );
        
    } while (f>tol || g >tol);
    
    a[0] = ra*cos(phi) + c[0];
    a[1] = ra*sin(phi) + c[1];

    point pa;
    pa.x = a[0]; pa.y = a[1]; pa.z = a[2];
    return pa;

}

void warp_refractive() {

    dim3 blocks(128,80,9);
    dim3 threads(10,10);

    warp_refractive_kernel<<<blocks, threads>>>();

    cudaDeviceSynchronize();

}

void gpu_calc_refocus_map(GpuMat &xmap, GpuMat &ymap, GpuMat &Hinv, GpuMat &P, GpuMat &PX, float z) {

    dim3 blocks(1, 1);
    dim3 threads(1, 1);

    //cudaMemcpyToSymbol()

    calc_refocus_map_kernel<<<blocks, threads>>>(xmap, ymap, Hinv, P, PX, z);

    cudaDeviceSynchronize();

}


