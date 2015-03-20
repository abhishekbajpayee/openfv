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

#include <fftw3.h>
//#include <cufftw.h>
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
    piv3D(int);
    
    void run(int);
    void add_frame(vector<Mat>);
    void batch_test();

protected:
    
private:
    
    vector<int> get_velocity_vector(double*, int, int, int, double&);
    void convolve3D(double*, double*, double*&, int, int, int);
    void convolve3D(double*, double*, double*&, int, int, int, fftw_plan, fftw_plan);
    void convolve3D(fftw_complex*, fftw_complex*, fftw_complex*&, int, int, int);

    vector< vector<int> > get_windows(int, int, double);

    // Math
    void multiply(fftw_complex*, fftw_complex*, fftw_complex*&, int n);
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
    
    // Flags
    int mean_shift_;
    int zero_padding_;

};

#endif
