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

#include "std_include.h"
#include "calibration.h"
#include "typedefs.h"
#include "piv.h"
// #include "cuda_lib.h"

//#include <fftw3.h>
#include <cufftw.h>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace std;
using namespace cv;

piv3D::piv3D(int l) {

    double *i1, *i2, *i3; 

    i1 = new double[l*l*l]; 
    i2 = new double[l*l*l];
    i3 = new double[l*l*l];

    for (int k = 0; k < l; k++) {
        for (int i = 0; i < l; i++) {
            for (int j = 0; j < l; j++) {         
                i1[k+l*(j+l*i)] = i+j+k;
                i2[k+l*(j+l*i)] = i+j+k+5;
            }
        }
    }

    double s = omp_get_wtime();
    convolve3D(i1, i2, i3, l, l, l); 
    LOG(INFO)<<omp_get_wtime()-s;

}

void piv3D::get_velocity_vector(double *a) {

    

}

void piv3D::convolve3D(fftw_complex *a, fftw_complex *b, fftw_complex* &out, int x, int y, int z) {

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
    multiply(o1, o2, o3, n);

    // Inverse FFT
    fftw_execute_dft(c2r, o3, out);
    normalize(out, n);

    fftw_destroy_plan(r2c); fftw_destroy_plan(c2r);

}

void piv3D::convolve3D(double *a, double *b, double* &out, int x, int y, int z) {

    int n = x * y * (z/2 +1);
    fftw_complex *o1, *o2, *o3;
    o1 = new fftw_complex[n]; o2 = new fftw_complex[n]; o3 = new fftw_complex[n];

    fftw_plan r2c, c2r;
    r2c = fftw_plan_dft_r2c_3d(x, y, z, a, o1, FFTW_ESTIMATE);
    c2r = fftw_plan_dft_c2r_3d(x, y, z, o3, out, FFTW_ESTIMATE);

    // FFT steps
    fftw_execute_dft_r2c(r2c, a, o1);
    fftw_execute_dft_r2c(r2c, b, o2);

    // Multiply the FFTs
    multiply(o1, o2, o3, n);

    // Inverse FFT
    fftw_execute_dft_c2r(c2r, o3, out);
    normalize(out, x * y * z);

    fftw_destroy_plan(r2c); fftw_destroy_plan(c2r);

}

void piv3D::multiply(fftw_complex *a, fftw_complex *b, fftw_complex* &out, int n) {

    for (int i = 0; i < n; i++) {
        out[i][0] = a[i][0]*b[i][0] - a[i][1]*b[i][1];
        out[i][1] = a[i][0]*b[i][1] + a[i][1]*b[i][0];
    }

}

void piv3D::normalize(fftw_complex* &a, int n) {

    for (int i = 0; i < n; i++) {
        a[i][0] /= n; a[i][1] /= n;
    }

}

void piv3D::normalize(double* &a, int n) {

    for (int i = 0; i < n; i++)
        a[i] /= n;

}

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
