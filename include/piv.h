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

//#include <fftw3.h>
#include <cufftw.h>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace std;
using namespace cv;

class piv3D {
    
public:
    ~piv3D() {}
    piv3D(int);
    
protected:
    
private:
    
    void get_velocity_vector(double *a);
    void convolve3D(double*, double*, double*&, int, int, int);
    void convolve3D(fftw_complex*, fftw_complex*, fftw_complex*&, int, int, int);

    // Math
    void multiply(fftw_complex*, fftw_complex*, fftw_complex*&, int n);
    void normalize(fftw_complex*&, int);
    void normalize(double*&, int);

    void print3D(double*, int, int, int);
    void print3D(fftw_complex*, int, int, int);
    
};

#endif
