// -------------------------------------------------------
// -------------------------------------------------------
// Synthetic Aperture - Particle Tracking Velocimetry Code
// --- Rendering Library Header ---
// -------------------------------------------------------
// Author: Abhishek Bajpayee
//         Dept. of Mechanical Engineering
//         Massachusetts Institute of Technology
// -------------------------------------------------------
// -------------------------------------------------------

#ifndef RENDERING_LIBRARY
#define RENDERING_LIBRARY

#include "std_include.h"
//#include "calibration.h"
#include "typedefs.h"
//#include "cuda_lib.h"

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace std;
using namespace cv;

class Scene {

 public:
    ~Scene() {

    }

 Scene() {}

    void seedParticles(vector< vector<double> > locations);
    void seedParticles(int num);

    // extents and bounds
    void create(double sx, int vx, double sy, int vy, double sz, int vz);

    // Note: the mem efficient way is to store x,y,z coords and intensity as a list
    // and not init a matrix of intensities. Too many black pixels...

    Mat getImg(int zv);

 private:

    void createVolume();
    vector<voxel> getVoxels(int z);

    double f(double x, double y, double z);

    vector<double> xlims_, ylims_, zlims_;
    double sx_, sy_, sz_;
    int vx_, vy_, vz_;
    vector<double> voxelsX_, voxelsY_, voxelsZ_;

    vector< vector<double> > particles_;
    vector<voxel> volume_;

};

class Camera {

 public:
    ~Camera() {

    }

 Camera() {}

    void init(double f, int imx, int imy);

    void bindScene(Scene scene);
    void setLocation(double x, double y, double z);
    void pointAt(double x, double y, double z);
    Mat render();

    Mat getP();

 private:
    
    Mat loc_;
    Mat t_;
    Mat R_;
    Mat K_;
    Mat P_;

    Scene scene_;

    //private members here

};


#endif
