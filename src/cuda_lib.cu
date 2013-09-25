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
#include <iostream>
#include <omp.h>

#include "cuda_lib.h"

using namespace cv;
using namespace gpu;

__global__ void add_kernel(float* a, float* b, float* c, float* exp) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int n=0; n<10; n++)
        c[i] = __powf( sqrt(a[i] + b[i]), exp[n] );

}

__global__ void calc_refocus_map_kernel(DevMem2Df xmap, DevMem2Df ymap, DevMem2Df Hinv, DevMem2Df P, DevMem2Df PX, DevMem2Df geom, float z) {

    /*

      NOTES:
      Can Hinv be applied even before entering CUDA part?
      Read initial points from global coalesced memory from array

    */

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    point p;
    p.x = Hinv.ptr(0)[0]*j + Hinv.ptr(0)[1]*i + Hinv.ptr(0)[2];
    p.y = Hinv.ptr(1)[0]*j + Hinv.ptr(1)[1]*i + Hinv.ptr(1)[2];
    p.z = z;
    
    point Xcam;
    Xcam.x = PX.ptr(0)[0]; Xcam.y = PX.ptr(1)[0]; Xcam.z = PX.ptr(2)[0];
    
    float zW = geom.ptr(0)[0];
    float n1 = geom.ptr(0)[1];
    float n2 = geom.ptr(0)[2];
    float n3 = geom.ptr(0)[3];
    float t = geom.ptr(0)[4];

    float f, g;
    point a = point_refrac_fast(Xcam, p, f, g, zW, n1, n2, n3, t);
    
    xmap.ptr(i)[j] = (P.ptr(0)[0]*a.x + P.ptr(0)[1]*a.y + P.ptr(0)[2]*a.z + P.ptr(0)[3])/
        (P.ptr(2)[0]*a.x + P.ptr(2)[1]*a.y + P.ptr(2)[2]*a.z + P.ptr(2)[3]);
    ymap.ptr(i)[j] = (P.ptr(1)[0]*a.x + P.ptr(1)[1]*a.y + P.ptr(1)[2]*a.z + P.ptr(1)[3])/
        (P.ptr(2)[0]*a.x + P.ptr(2)[1]*a.y + P.ptr(2)[2]*a.z + P.ptr(2)[3]);

    //printf("residuals %f, %f\n", f, g);

}

__device__ point point_refrac(point Xcam, point p, float &f, float &g, float zW_, float n1_, float n2_, float n3_, float t_) {
 
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
        
    float rp = sqrt( (pt[0]-c[0])*(pt[0]-c[0]) + (pt[1]-c[1])*(pt[1]-c[1]) );
    float dp = pt[2]-b[2];
    float phi = atan2(pt[1]-c[1],pt[0]-c[0]);
    
    float ra = sqrt( (a[0]-c[0])*(a[0]-c[0]) + (a[1]-c[1])*(a[1]-c[1]) );
    float rb = sqrt( (b[0]-c[0])*(b[0]-c[0]) + (b[1]-c[1])*(b[1]-c[1]) );
    float da = a[2]-c[2];
    float db = b[2]-a[2];
    
    //float f, g; 
    float dfdra, dfdrb, dgdra, dgdrb;
        
    // Newton Raphson loop to solve for Snell's law
    for (int i=0; i<10; i++) {
        
        f = ( ra/sqrt((ra*ra)+(da*da)) ) - ( (n2_/n1_)*(rb-ra)/sqrt((rb-ra)*(rb-ra) + (db*db)) );
        g = ( (rb-ra)/sqrt((rb-ra)*(rb-ra)+(db*db)) ) - ( (n3_/n2_)*(rp-rb)/sqrt((rp-rb)*(rp-rb)+(dp*dp)) );
        
        dfdra = ( 1.0/sqrt((ra*ra)+(da*da)) )
            - ( (ra*ra)/powf((ra*ra)+(da*da),1.5) )
            + ( (n2_/n1_)/sqrt((ra-rb)*(ra-rb)+(db*db)) )
            - ( (n2_/n1_)*(ra-rb)*(2*ra-2*rb)/(2*powf((ra-rb)*(ra-rb)+(db*db),1.5)) );
        
        dfdrb = ( (n2_/n1_)*(ra-rb)*(2*ra-2*rb)/(2*powf((ra-rb)*(ra-rb)+(db*db),1.5)) )
            - ( (n2_/n1_)/sqrt((ra-rb)*(ra-rb)+(db*db)) );
        
        dgdra = ( (ra-rb)*(2*ra-2*rb)/(2*powf((ra-rb)*(ra-rb)+(db*db),1.5)) )
            - ( (1.0)/sqrt((ra-rb)*(ra-rb)+(db*db)) );
        
        dgdrb = ( (1.0)/sqrt((ra-rb)*(ra-rb)+(db*db)) )
            + ( (n3_/n2_)/sqrt((rb-rp)*(rb-rp)+(dp*dp)) )
            - ( (ra-rb)*(2*ra-2*rb)/(2*powf((ra-rb)*(ra-rb)+(db*db),1.5)) )
            - ( (n3_/n2_)*(rb-rp)*(2*rb-2*rp)/(2*powf((rb-rp)*(rb-rp)+(dp*dp),1.5)) );
        
        ra = ra - ( (f*dgdrb - g*dfdrb)/(dfdra*dgdrb - dfdrb*dgdra) );
        rb = rb - ( (g*dfdra - f*dgdra)/(dfdra*dgdrb - dfdrb*dgdra) );
        
    }

    a[0] = ra*cos(phi) + c[0];
    a[1] = ra*sin(phi) + c[1];

    point pa;
    pa.x = a[0]; pa.y = a[1]; pa.z = a[2];
    return pa;

}

__device__ point point_refrac_fast(point Xcam, point p, float &f, float &g, float zW_, float n1_, float n2_, float n3_, float t_) {

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
        
    float rp = sqrt( (pt[0]-c[0])*(pt[0]-c[0]) + (pt[1]-c[1])*(pt[1]-c[1]) );
    float dp = pt[2]-b[2];
    float phi = atan2(pt[1]-c[1],pt[0]-c[0]);
    
    float ra = sqrt( (a[0]-c[0])*(a[0]-c[0]) + (a[1]-c[1])*(a[1]-c[1]) );
    float rb = sqrt( (b[0]-c[0])*(b[0]-c[0]) + (b[1]-c[1])*(b[1]-c[1]) );
    float da = a[2]-c[2];
    float db = b[2]-a[2];
    
    //float f, g; 
    float dfdra, dfdrb, dgdra, dgdrb;
        
    // Newton Raphson loop to solve for Snell's law
    for (int i=0; i<10; i++) {
        
        f = ( ra/sqrt((ra*ra)+(da*da)) ) - ( (n2_/n1_)*(rb-ra)/sqrt((rb-ra)*(rb-ra) + (db*db)) );
        g = ( (rb-ra)/sqrt((rb-ra)*(rb-ra)+(db*db)) ) - ( (n3_/n2_)*(rp-rb)/sqrt((rp-rb)*(rp-rb)+(dp*dp)) );
        
        dfdra = ( 1.0/sqrt((ra*ra)+(da*da)) )
            - ( (ra*ra)/__powf((ra*ra)+(da*da),1.5) )
            + ( (n2_/n1_)/sqrt((ra-rb)*(ra-rb)+(db*db)) )
            - ( (n2_/n1_)*(ra-rb)*(2*ra-2*rb)/(2*__powf((ra-rb)*(ra-rb)+(db*db),1.5)) );
        
        dfdrb = ( (n2_/n1_)*(ra-rb)*(2*ra-2*rb)/(2*__powf((ra-rb)*(ra-rb)+(db*db),1.5)) )
            - ( (n2_/n1_)/sqrt((ra-rb)*(ra-rb)+(db*db)) );
        
        dgdra = ( (ra-rb)*(2*ra-2*rb)/(2*__powf((ra-rb)*(ra-rb)+(db*db),1.5)) )
            - ( (1.0)/sqrt((ra-rb)*(ra-rb)+(db*db)) );
        
        dgdrb = ( (1.0)/sqrt((ra-rb)*(ra-rb)+(db*db)) )
            + ( (n3_/n2_)/sqrt((rb-rp)*(rb-rp)+(dp*dp)) )
            - ( (ra-rb)*(2*ra-2*rb)/(2*__powf((ra-rb)*(ra-rb)+(db*db),1.5)) )
            - ( (n3_/n2_)*(rb-rp)*(2*rb-2*rp)/(2*__powf((rb-rp)*(rb-rp)+(dp*dp),1.5)) );
        
        ra = ra - ( (f*dgdrb - g*dfdrb)/(dfdra*dgdrb - dfdrb*dgdra) );
        rb = rb - ( (g*dfdra - f*dgdra)/(dfdra*dgdrb - dfdrb*dgdra) );
        
    }

    a[0] = ra*cos(phi) + c[0];
    a[1] = ra*sin(phi) + c[1];

    point pa;
    pa.x = a[0]; pa.y = a[1]; pa.z = a[2];
    return pa;

}

void gpu_calc_refocus_map(GpuMat &xmap, GpuMat &ymap, GpuMat &Hinv, GpuMat &P, GpuMat &PX, GpuMat &geom, float z) {

    dim3 blocks(40, 25);
    dim3 threads(32, 32);

    double wall = omp_get_wtime();
    calc_refocus_map_kernel<<<blocks, threads>>>(xmap, ymap, Hinv, P, PX, geom, z);
    cudaDeviceSynchronize();
    //std::cout<<cudaGetErrorString(cudaGetLastError())<<std::endl;
    std::cout<<"Time: "<<omp_get_wtime()-wall<<std::endl;

}

void add() {

    int n=1;

    int num = 1000000;
    size_t size = num*sizeof(float);

    float *a, *b, *h_exp;
    float *d_a, *d_b, *d_c, *exp;

    cudaMalloc(&exp, 10*sizeof(float));
    h_exp = new float[10];
    for (int i=0; i<10; i++)
        h_exp[i] = 3;
    cudaMemcpy(exp, h_exp, 10*sizeof(float), cudaMemcpyHostToDevice);

    if (n) {
        
        a = new float[num];
        b = new float[num];
        for (int i=0; i<num; i++) {
            a[i] = 2;
            b[i] = 3;
        }

        cudaMalloc(&d_a, size); 
        cudaMalloc(&d_b, size);
        cudaMalloc(&d_c, size);
        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
        
        double wall = omp_get_wtime();
        add_kernel<<<1000, 1000>>>(d_a, d_b, d_c, exp);
        cudaDeviceSynchronize();
        std::cout<<"Time: "<<omp_get_wtime()-wall<<std::endl;

    } else {

        a = new float[1];
        b = new float[1];
        a[0] = 2; b[0] = 3;

        size = sizeof(float);
        cudaMalloc(&d_a, size); 
        cudaMalloc(&d_b, size);
        cudaMalloc(&d_c, size);
        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

        double wall = omp_get_wtime();
        add_kernel<<<1, 1>>>(d_a, d_b, d_c, exp);
        cudaDeviceSynchronize();
        std::cout<<"Time: "<<omp_get_wtime()-wall<<std::endl;

    }


}

