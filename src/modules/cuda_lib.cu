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

#include "cuda_lib.h"

// Constant variables on device
__constant__ float Hinv[6];
__constant__ float PX[9][3];
__constant__ float P[9][12];
__constant__ float zW_;
__constant__ float n1_;
__constant__ float n2_;
__constant__ float n3_;
__constant__ float t_;


__global__ void calc_refocus_map_kernel(PtrStepSzf xmap, PtrStepSzf ymap, float z, int n, int rows, int cols) {

    /*

      NOTES:
      Can Hinv be applied even before entering CUDA part?
      Read initial points from global coalesced memory from array

    */

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < cols && i < rows) {

        point p;
        p.x = Hinv[0]*j + Hinv[1]*i + Hinv[2];
        p.y = Hinv[3]*j + Hinv[4]*i + Hinv[5];
        p.z = z;

        point Xcam;
        Xcam.x = PX[n][0]; Xcam.y = PX[n][1]; Xcam.z = PX[n][2];

        float f, g;
        point a = point_refrac_fast(Xcam, p, f, g);

        xmap.ptr(i)[j] = (P[n][0]*a.x + P[n][1]*a.y + P[n][2]*a.z + P[n][3])/(P[n][8]*a.x + P[n][9]*a.y + P[n][10]*a.z + P[n][11]);
        ymap.ptr(i)[j] = (P[n][4]*a.x + P[n][5]*a.y + P[n][6]*a.z + P[n][7])/(P[n][8]*a.x + P[n][9]*a.y + P[n][10]*a.z + P[n][11]);

        //printf("residuals %f, %f\n", f, g);

    }

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

    float dfdra, dfdrb, dgdra, dgdrb;

    float tol = 1E-9;
    float ra1, rb1, res;
    ra1 = ra; rb1 = rb;

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

        res = abs(ra1-ra)+abs(rb1-rb);
        ra1 = ra; rb1 = rb;
        if (res < tol)
            break;

    }

    a[0] = ra*cos(phi) + c[0];
    a[1] = ra*sin(phi) + c[1];

    point pa;
    pa.x = a[0]; pa.y = a[1]; pa.z = a[2];
    return pa;

}

__device__ point point_refrac_fast(point Xcam, point p, float &f, float &g) {

    float c[3];
    c[0] = Xcam.x; c[1] = Xcam.y; c[2] = Xcam.z;

    float a[3];
    float b[3];
    float pt[3];
    pt[0] = p.x; pt[1] = p.y; pt[2] = p.z;

    float ptx = pt[0]-c[0]; float pty = pt[1]-c[1]; float ptz = pt[2]-c[2];
    float zWz = zW_-c[2];

    a[0] = c[0] + (ptx*zWz)/ptz; a[1] = c[1] + (pty*zWz)/ptz; a[2] = zW_;
    b[0] = c[0] + (ptx*zWz)/ptz; b[1] = c[1] + (pty*zWz)/ptz; b[2] = t_+zW_;

    float rp = sqrt( (ptx*ptx) + (pty*pty) );
    float dp = pt[2]-b[2];
    float phi = atan2(pty,ptx);

    float acx = a[0]-c[0]; float acy = a[1]-c[1]; float acz = a[2]-c[2];

    float ra = sqrt( (acx*acx) + (acy*acy) );
    float rb = sqrt( (b[0]-c[0])*(b[0]-c[0]) + (b[1]-c[1])*(b[1]-c[1]) ); // TODO
    float da = acz;
    float db = b[2]-a[2]; // TODO

    float dfdra, dfdrb, dgdra, dgdrb;
    float rasq, dasq, dbsq, dpsq, rbra, rbrasq, rprb, rprbsq;
    float n2n1 = n2_/n1_; float n3n2 = n3_/n2_;
    dasq = da*da; dbsq = db*db; dpsq = dp*dp;

    float tol = 1E-9;
    float ra1, rb1, res;
    ra1 = ra; rb1 = rb;

    // Newton Raphson loop to solve for Snell's law
    for (int i=0; i<10; i++) {

        rasq = ra*ra;
        rbra = rb-ra; rbrasq = rbra*rbra;
        rprb = rp-rb; rprbsq = rprb*rprb;

        f = ( ra/sqrt(rasq+dasq) ) - ( (n2n1)*(rbra)/sqrt(rbrasq+dbsq) );
        g = ( rbra/sqrt(rbrasq+dbsq) ) - ( (n3n2*rprb)/sqrt(rprbsq+dpsq) );

        dfdra = ( 1.0/sqrt(rasq+dasq) )
            - ( rasq/__powf(rasq+dasq,1.5) )
            + ( n2n1/sqrt(rbrasq+dbsq) )
            - ( (n2n1*-rbra)*(2*ra-2*rb)/(2*__powf(rbrasq+dbsq,1.5)) );

        dfdrb = ( (n2n1*-rbra)*(2*ra-2*rb)/(2*__powf(rbrasq+dbsq,1.5)) )
            - ( n2n1/sqrt(rbrasq+dbsq) );

        dgdra = ( -rbra*(2*ra-2*rb)/(2*__powf(rbrasq+dbsq,1.5)) )
            - ( 1.0/sqrt(rbrasq+dbsq) );

        dgdrb = ( 1.0/sqrt(rbrasq+dbsq) )
            + ( n3n2/sqrt(rprbsq+dpsq) )
            - ( -rbra*(2*ra-2*rb)/(2*__powf(rbrasq+dbsq,1.5)) )
            - ( (n3n2*-rprb)*(2*rb-2*rp)/(2*__powf(rprbsq+dpsq,1.5)) );

        ra = ra - ( (f*dgdrb - g*dfdrb)/(dfdra*dgdrb - dfdrb*dgdra) );
        rb = rb - ( (g*dfdra - f*dgdra)/(dfdra*dgdrb - dfdrb*dgdra) );

        res = abs(ra1-ra)+abs(rb1-rb);
        ra1 = ra; rb1 = rb;
        if (res < tol)
            break;

    }

    a[0] = ra*cos(phi) + c[0];
    a[1] = ra*sin(phi) + c[1];

    point pa;
    pa.x = a[0]; pa.y = a[1]; pa.z = a[2];
    return pa;

}

__global__ void calc_nlca_image_fast(PtrStepSzf nlca_image, PtrStepSzf img1, PtrStepSzf img2, PtrStepSzf img3, PtrStepSzf img4, int rows, int cols, float sigma) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < cols && i < rows) {

        float v1 = img1.ptr(i)[j];
        __syncthreads();
        float v2 = img2.ptr(i)[j];
        __syncthreads();
        float v3 = img3.ptr(i)[j];
        __syncthreads();
        float v4 = img4.ptr(i)[j];
        __syncthreads();

        v1 = (v1 > 1.0) ? 1.0 : v1;
        v2 = (v2 > 1.0) ? 1.0 : v2;
        v3 = (v3 > 1.0) ? 1.0 : v3;
        v4 = (v4 > 1.0) ? 1.0 : v4;

        float mean = 0.25*(v1+v2+v3+v4);

        float out_val = exp( -0.5*( ((mean-1.0)/sigma) * ((mean-1.0)/sigma) ) );

        nlca_image.ptr(i)[j] = out_val;

    }

}

__global__ void calc_nlca_image(PtrStepSzf nlca_image, PtrStepSzf img1, PtrStepSzf img2, PtrStepSzf img3, PtrStepSzf img4, int rows, int cols, int window, float sigma) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    int tj = threadIdx.x;
    int ti = threadIdx.y;

    extern __shared__ float win[];
    __shared__ float wmin;
    __shared__ float wmax;

    if (j < cols && i < rows) {

        // pre clip values over 1?

        float v1 = img1.ptr(i)[j];
        __syncthreads();
        float v2 = img2.ptr(i)[j];
        __syncthreads();
        float v3 = img3.ptr(i)[j];
        __syncthreads();
        float v4 = img4.ptr(i)[j];
        __syncthreads();

        v1 = (v1 > 1.0) ? 1.0 : v1;
        v2 = (v2 > 1.0) ? 1.0 : v2;
        v3 = (v3 > 1.0) ? 1.0 : v3;
        v4 = (v4 > 1.0) ? 1.0 : v4;

        float mean = 0.25*(v1+v2+v3+v4);

        win[tj*window+ti] = mean;
        __syncthreads();

        // calculate max only once in block
        if (tj==0 && ti == 0) {
            wmax = 0.0;
            wmin = 1.0;
            for (int id=0; id<window*window; id++) {
                if (win[id] > wmax)
                    wmax = win[id];
                if (win[id] < wmin)
                    wmin = win[id];
            }
        }
        __syncthreads();

        float out_val = exp( -0.5*( ((mean-wmax)/sigma) * ((mean-wmax)/sigma) ) );

        nlca_image.ptr(i)[j] = out_val;

    }

}

void uploadRefractiveData(float hinv[6], float locations[9][3], float pmats[9][12], float geom[5]) {

    cudaMemcpyToSymbol(Hinv, hinv, sizeof(float)*6);

    cudaMemcpyToSymbol(zW_, &geom[0], sizeof(float));
    cudaMemcpyToSymbol(n1_, &geom[1], sizeof(float));
    cudaMemcpyToSymbol(n2_, &geom[2], sizeof(float));
    cudaMemcpyToSymbol(n3_, &geom[3], sizeof(float));
    cudaMemcpyToSymbol(t_, &geom[4], sizeof(float));

    cudaMemcpyToSymbol(PX, locations, 9*3*sizeof(float));
    cudaMemcpyToSymbol(P, pmats, 9*12*sizeof(float));

}

void gpu_calc_refocus_map(GpuMat &xmap, GpuMat &ymap, float z, int i, int rows, int cols) {

    dim3 block(32, 32);
    dim3 grid(ceil(cols/16), ceil(rows/16));

    // dim3 grid(80, 50); dim3 block(16, 16);
    // dim3 grid(50, 50); dim3 block(10, 10);

    if (!cudaGetLastError()) {
        calc_refocus_map_kernel<<<grid, block>>>(xmap, ymap, z, i, rows, cols);
    } else {
        std::cout<<cudaGetErrorString(cudaGetLastError())<<std::endl;
    }
    cudaDeviceSynchronize();

}

// TODO: only works for 4 cameras!!!
void gpu_calc_nlca_image_fast(vector<GpuMat> &warped, GpuMat &nlca_image, int rows, int cols, float sigma) {
    
    dim3 block(32, 32);
    dim3 grid(ceil(float(cols)/32.0), ceil(float(rows)/32.0));

    if (!cudaGetLastError()) {
        calc_nlca_image_fast<<<grid, block>>>(nlca_image, warped[0], warped[1], warped[2], warped[3], rows, cols, sigma);
    } else {
        std::cout<<cudaGetErrorString(cudaGetLastError())<<std::endl;
    }
    cudaDeviceSynchronize();

}

// TODO: only works for 4 cameras!!!
void gpu_calc_nlca_image(vector<GpuMat> &warped, GpuMat &nlca_image, int rows, int cols, int window, float sigma) {
    
    dim3 block(window, window);
    dim3 grid(ceil(float(cols)/float(window)), ceil(float(rows)/float(window)));

    if (!cudaGetLastError()) {
        calc_nlca_image<<<grid, block, window*window*sizeof(float)>>>(nlca_image, warped[0], warped[1], warped[2], warped[3], rows, cols, window, sigma);
    } else {
        std::cout<<cudaGetErrorString(cudaGetLastError())<<std::endl;
    }

    cudaDeviceSynchronize();

}
