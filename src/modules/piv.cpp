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

// -------------------------------------------------------
// -------------------------------------------------------
// Synthetic Aperture - Particle Tracking Velocimetry Code
// --- 3D PIV Library ---
// -------------------------------------------------------
// Author: Abhishek Bajpayee
//         Dept. of Mechanical Engineering
//         Massachusetts Institute of Technology
// -------------------------------------------------------
// -------------------------------------------------------

#include "piv.h"
#include "tools.h"

#include <cufftw.h>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace std;
using namespace cv;

piv3D::piv3D(string resultFile) {

    frames_ = 0;

    // Mean shift ON by default for now. TODO: Find out if this makes a difference.
    mean_shift_ = 1;

    // Zero padding ON by default
    zero_padding_ = 1;

    resultFile_ = resultFile;

}

void piv3D::add_frame(vector<Mat> mats) {

    VLOG(1)<<"Adding frame...";

    Mat3 vol(mats);
    frames.push_back(vol);
    if (frames_==0) {

        zs_ = mats.size();

        if (zs_ == 0)
            LOG(FATAL)<<"Empty volume!";

        xs_ = mats[0].rows;
        ys_ = mats[0].cols;

    }

    frames_++;

    VLOG(1)<<"Frames now: "<<frames_;

}

void piv3D::run(int l, double overlap) {

    wx_ = l; wy_ = l; wz_ = l;

    vector< vector<int> > winx, winy, winz;
    winx = get_windows(xs_, wx_, overlap);
    winy = get_windows(ys_, wy_, overlap);
    winz = get_windows(zs_, wz_, overlap);

    if (zero_padding_) {
        wx_ *= 2; wy_ *= 2; wz_ *= 2;
    }

    run_pass(winx, winy, winz);

}

void piv3D::run_pass(vector< vector<int> > winx, vector< vector<int> > winy, vector< vector<int> > winz) {

    double *i1, *i2, *i3;
    fftw_complex *o1;
    i1 = new double[wx_*wx_*wx_];
    i2 = new double[wy_*wy_*wy_];
    i3 = new double[wz_*wz_*wz_];

    // double s = omp_get_wtime();
    
    fileIO file(resultFile_);

    int count=0;
    //#pragma omp parallel for
    for (int i = 0; i < winx.size(); i++) {
        for (int j = 0; j < winy.size(); j++) {
            for (int k = 0; k < winz.size(); k++) {
                
           
                frames[0].getWindow(winx[i][0], winx[i][1], winy[j][0], winy[j][1], winz[k][0], winz[k][1], i1, zero_padding_);
                frames[1].getWindow(winx[i][0], winx[i][1], winy[j][0], winy[j][1], winz[k][0], winz[k][1], i2, zero_padding_);
                crossex3D(i1, i2, i3, wx_, wy_, wz_);

                vector<int> mloc; double val;
                mloc = get_velocity_vector(i3, wx_, wy_, wz_, val);
                
                VLOG(1)<<"["<<winx[i][0]<<", "<<winx[i][1]<<"], ["<<winy[j][0]<<", "<<winy[j][1]<<"], ["<<winz[k][0]<<", "<<winz[k][1]<<"]: "<<mloc[0]<<", "<<mloc[1]<<", "<<mloc[2];
                count++;

                file<<(winx[i][0]+winx[i][1])/2<<"\t"<<(winy[j][0]+winy[j][1])/2<<"\t"<<(winz[k][0]+winz[k][1])/2<<"\t"<<mloc[0]<<"\t"<<mloc[1]<<"\t"<<mloc[2]<<"\n";

            }
        }
    }

    // double time = omp_get_wtime()-s;

    // print3D(i1, l, l, l);
    // print3D(i2, l, l, l);
    // print3D(i3, l, l, l);

    // LOG(INFO)<<time<<", "<<time/count;

}

// TODO: z velocity is incorrect and needs to be handled
vector<int> piv3D::get_velocity_vector(double *a, int x, int y, int z, double &maxval) {

    maxval=0;
    int mx, my, mz;
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            for (int k = 0; k < z; k++) {
                // int ind = k+y*(j+z*i);
                int ind = k+y*(j+x*i);
                if (a[ind]>maxval) {
                    maxval = a[ind];
                    mx = i; my = j; mz = k;
                }
            }
        }
    }

    // Modding and shifting
    mx = x/2 - (mx + x/2)%x; my = y/2 - (my + y/2)%y; mz = z/2 - (mz + z/2)%z;

    // Shifting vector around center
    // mx -= (x-1)/2; my -= (y-1)/2; mz -= (z-1)/2;

    vector<int> vector;
    vector.push_back(mx); vector.push_back(my); vector.push_back(mz);
    return(vector);

}

// Cross correlation function for complex inputs and outputs
void piv3D::crossex3D(fftw_complex *a, fftw_complex *b, fftw_complex* &out, int x, int y, int z) {

    int n = x * y * z;
    fftw_complex *o1, *o2, *o3;
    o1 = new fftw_complex[n]; o2 = new fftw_complex[n]; o3 = new fftw_complex[n];

    fftw_plan r2c, c2r;
    r2c = fftw_plan_dft_3d(x, y, z, a, o1, -1, FFTW_ESTIMATE);
    c2r = fftw_plan_dft_3d(x, y, z, o3, out, 1, FFTW_ESTIMATE);

    // FFT steps
    fftw_execute_dft(r2c, a, o1);
    fftw_execute_dft(r2c, b, o2);

    // Multiply the FFTs
    multiply_conjugate(o1, o2, o3, n);

    // Inverse FFT
    fftw_execute_dft(c2r, o3, out);
    normalize(out, n);

    fftw_destroy_plan(r2c); fftw_destroy_plan(c2r);

}

// Cross correlation function for real inputs and output
// NOTE: The cross correlation function returns a result that is shifted by half
// the window size so needs to be taken care of when peak is calculated to find
// velocity. TODO: Look into whether this is convention or not.
void piv3D::crossex3D(double *a, double *b, double* &out, int x, int y, int z) {

    int n = x * y * (z/2 +1);
    fftw_complex *o1, *o2, *o3;
    o1 = new fftw_complex[n]; o2 = new fftw_complex[n]; o3 = new fftw_complex[n];

    fftw_plan r2c, c2r;
    r2c = fftw_plan_dft_r2c_3d(x, y, z, a, o1, FFTW_ESTIMATE);
    c2r = fftw_plan_dft_c2r_3d(x, y, z, o3, out, FFTW_ESTIMATE);

    if (mean_shift_) {
        mean_shift(a, x*y*z);
        mean_shift(b, x*y*z);
    }

    // FFT steps
    fftw_execute_dft_r2c(r2c, a, o1);
    fftw_execute_dft_r2c(r2c, b, o2);

    // Multiply the FFTs
    multiply_conjugate(o1, o2, o3, n);

    // Inverse FFT
    fftw_execute_dft_c2r(c2r, o3, out);
    normalize(out, x * y * z);

    fftw_destroy_plan(r2c); fftw_destroy_plan(c2r);
    delete [] o1; delete [] o2; delete [] o3;

}

// Cross correlation function for real inputs and output based on plan (use for batch fft)
void piv3D::crossex3D(double *a, double *b, double* &out, int x, int y, int z, fftw_plan r2c, fftw_plan c2r) {

    int n = x * y * (z/2 +1);
    fftw_complex *o1, *o2, *o3;
    o1 = new fftw_complex[n]; o2 = new fftw_complex[n]; o3 = new fftw_complex[n];

    if (mean_shift_) {
        mean_shift(a, x*y*z);
        mean_shift(b, x*y*z);
    }

    // FFT steps
    fftw_execute_dft_r2c(r2c, a, o1);
    fftw_execute_dft_r2c(r2c, b, o2);

    // Multiply the FFTs
    multiply_conjugate(o1, o2, o3, n);

    // Inverse FFT
    fftw_execute_dft_c2r(c2r, o3, out);
    normalize(out, x * y * z);

    delete [] o1;
    delete [] o2;
    delete [] o3;

}

// Multiply complex volume 1 element wise with complex conjugate of complex volume 2
void piv3D::multiply_conjugate(fftw_complex *a, fftw_complex *b, fftw_complex*& out, int n) {

    // multiplying the complex conjugate of a with b
    for (int i = 0; i < n; i++) {
        out[i][0] = a[i][0]*b[i][0] + a[i][1]*b[i][1];
        out[i][1] = a[i][0]*b[i][1] - a[i][1]*b[i][0];
    }

}

// Normalize complex volume
void piv3D::normalize(fftw_complex*& a, int n) {

    for (int i = 0; i < n; i++) {
        a[i][0] /= n; a[i][1] /= n;
    }

}

// Normalize real volume
void piv3D::normalize(double*& a, int n) {

    for (int i = 0; i < n; i++)
        a[i] /= n;

}

// Output real volume
void piv3D::print3D(double *a, int x, int y, int z) {

    for (int k = 0; k < z; k++) {
        cout<<"[";
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                cout<<a[k+y*(j+x*i)];
                if (j<y-1)
                    cout<<",\t";
                else
                    cout<<";";
            }
            if (i<x-1)
                cout<<"\n";
            else
                cout<<"]\n";
        }
    }

}

// Output complex volume
void piv3D::print3D(fftw_complex *a, int x, int y, int z) {

    for (int k = 0; k < z; k++) {
        
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                cout<<a[k+y*(j+x*i)][0]<<" + "<<a[k+y*(j+x*i)][1]<<"i,\t";
            }
            cout<<endl;
        }
        
    }

}

// Shift levels of real volume so mean is zero
void piv3D::mean_shift(double*& a, int n) {

    double sum = 0;

    for (int i = 0; i < n; i++)
        sum += a[i];
    
    sum /= n;

    for (int i = 0; i < n; i++)
        a[i] -= sum;
    

}

void piv3D::batch_test() {

    int N = 32;
    int n = 8;

    double *i1 = new double[N];
    for (int i = 0; i < N; i++) {
        i1[i] = i;
    }

    int num = 2;

    double *in = new double[n];
    fftw_complex *o1 = new fftw_complex[n/2+1];   

    fftw_plan plan1;
    plan1 = fftw_plan_dft_r2c_1d(n, in, o1, FFTW_ESTIMATE);

    cout<<"Input and output arrays from single window runs:"<<endl;

    for (int s = 0; s < 10; s++) {      
        for (int i = 0; i < n; i++) {
            in[i] = i1[i+s];
        } 
    
        fftw_execute_dft_r2c(plan1, in, o1);
    
        // for (int i = 0; i < n; i++) {
        //     cout<<in[i]<<"\t";
        // }
        // cout<<endl;
        for (int i = 0; i < n/2+1; i++) {
            cout<<o1[i][0]<<" + "<<o1[i][1]<<"j, ";
        }
        cout<<endl;
    }

    fftw_plan plan2;
    int idist = 1;
    int odist = n/2+1;
    int size[] = {n};
    fftw_complex *o2 = new fftw_complex[N];
    plan2 = fftw_plan_many_dft_r2c(1, size, num, i1, NULL, 1, idist, o2, NULL, 1, odist, FFTW_ESTIMATE);
    fftw_execute_dft_r2c(plan2, i1, o2);

    cout<<endl<<"Output from batched run:"<<endl;
    for (int s = 0; s < num; s++) {
        for (int i = 0; i < n/2+1; i++) {
            cout<<o2[i+s*(n/2+1)][0]<<" + "<<o2[i+s*(n/2+1)][1]<<"j, ";
        }
        cout<<endl;
    }
    cout<<endl;

}

// Generate list of window bounds given total size, window size
// and overlap
vector< vector<int> > piv3D::get_windows(int s, int w, double overlap) {

    vector< vector<int> > outer;
    vector<int> inner;

    int start = 0; int end = start+w-1;
    inner.push_back(start); inner.push_back(end);
    outer.push_back(inner); inner.clear();

    while (end<s-1) {
        start += w*(1-overlap);
        end += w*(1-overlap);
        inner.push_back(start); inner.push_back(end);
        outer.push_back(inner); inner.clear();
    }

    return(outer);

}

// Container to store stack of Mats as 3D volume
Mat3::Mat3(vector<Mat> volume): volume_(volume) {

}

// Return subvolume from Mat3 volume as pointer array
void Mat3::getWindow(int x1, int x2, int y1, int y2, int z1, int z2, double*& win, int zero_padding) {

    int nx = x2 - x1 + 1; int ny = y2 - y1 + 1; int nz = z2 - z1 + 1;
    int sx = 0; int sy = 0; int sz = 0;

    if (zero_padding) {
        sx = nx/2; sy = ny/2; sz = nz/2;
        nx *= 2; ny *= 2; nz *= 2;

        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    int ind = (k + ny*(j + nx*i));
                    win[ind] = 0;
                }
            }
        }

    }

    for (int i = x1; i <= x2; i++) {
        for (int j = y1; j <= y2; j++) {
            for (int k = z1; k <= z2; k++) {
                int ind = (k+sz-z1) + ny*((j+sy-y1) + nx*(i+sx-x1));
                win[ind] = volume_[k].at<float>(j,i);
            }
        }
    }

}
