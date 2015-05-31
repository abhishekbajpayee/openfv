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
// --- 3D PIV Library Header ---
// -------------------------------------------------------
// Author: Abhishek Bajpayee
//         Dept. of Mechanical Engineering
//         Massachusetts Institute of Technology
// -------------------------------------------------------
// -------------------------------------------------------

#ifndef PIV_LIBRARY
#define PIV_LIBRARY

#include "std_include.h"
#include "calibration.h"
#include "typedefs.h"
// #include "cuda_lib.h"

// #include <fftw3.h>// 
#include <cufftw.h>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace std;
using namespace cv;

class Mat3 {
    
public:
    ~Mat3() {}
    Mat3(vector<Mat>);
    
    void getWindow(int, int, int, int, int, int, double*&, int);

protected:
    
private:

    vector<Mat> volume_;

    // Flags

};

class piv3D {
    
public:
    ~piv3D() {}
    piv3D(string);
    
    void run(int, double);
    void add_frame(vector<Mat>);
    void batch_test();

protected:
    
private:

    void run_pass(vector< vector<int> >, vector< vector<int> >, vector< vector<int> >);
    
    vector<int> get_velocity_vector(double*, int, int, int, double&);
    void crossex3D(double*, double*, double*&, int, int, int);
    void crossex3D(double*, double*, double*&, int, int, int, fftw_plan, fftw_plan);
    void crossex3D(fftw_complex*, fftw_complex*, fftw_complex*&, int, int, int);

    vector< vector<int> > get_windows(int, int, double);

    // Math
    void multiply_conjugate(fftw_complex*, fftw_complex*, fftw_complex*&, int n);
    void normalize(fftw_complex*&, int);
    void normalize(double*&, int);
    void mean_shift(double*&, int);

    void print3D(double*, int, int, int);
    void print3D(fftw_complex*, int, int, int);

    vector<Mat3> frames;

    // number of frames in sequence
    int frames_;
    int xs_, ys_, zs_;
    int wx_, wy_, wz_;
    
    string resultFile_;

    // Flags
    int mean_shift_;
    int zero_padding_;

};

#endif
