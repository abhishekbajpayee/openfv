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
#include "typedefs.h"
#include "refocusing.h"
#include "serialization.h"

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace std;
using namespace cv;

class Scene {

 public:
    ~Scene() {

    }

    Scene();

    void create(double sx, double sy, double sz, int gpu);
    void setGpuFlag(int gpu);

    void renderVolume(int xv, int yv, int zv);
    void renderVolumeCPU(int xv, int yv, int zv);
    void renderVolumeGPU(int xv, int yv, int zv);
    void renderVolumeGPU2(int xv, int yv, int zv);

    void setParticleSigma(double, double, double);
    void setRefractiveGeom(float zW, float n1, float n2, float n3, float t);

    void seedR();
    void seedAxes();
    void seedParticles(vector< vector<double> > locations);
    void seedParticles(int num, double factor);    

    void propagateParticles(vector<double> (*func)(double, double, double, double), double t);

    Mat getSlice(int zv);

    Mat getParticles();
    vector<float> getRefGeom();
    int getRefFlag();
    vector<int> getVoxelGeom();
    vector<double> getSceneGeom();
    double sigma();

    void temp();

    void dumpStack(string);

  private:

    // Function to serialize and save Scene object
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & sigmax_ & sigmay_ & sigmaz_;
        ar & xlims_ & ylims_ & zlims_;
        ar & sx_ & sy_ & sz_;
        ar & vx_ & vy_ & vz_;
        ar & voxelsX_, voxelsY_, voxelsZ_;
        ar & particles_;
        ar & trajectory_;
        ar & volumeGPU_;
        ar & volumeCPU_;
        ar & GPU_FLAG;
        ar & REF_FLAG;
        ar & geom_;
    }

    double f(double x, double y, double z);

    double sigmax_, sigmay_, sigmaz_;
    vector<double> xlims_, ylims_, zlims_;
    double sx_, sy_, sz_;
    int vx_, vy_, vz_;
    vector<double> voxelsX_, voxelsY_, voxelsZ_;

    Mat_<double> particles_;
    vector< Mat_<double> > trajectory_;
    vector<Mat> volumeGPU_;
    vector<Mat> volumeCPU_;

    gpu::GpuMat gx, gy;
    gpu::GpuMat tmp1, tmp2, tmp3, tmp4;
    gpu::GpuMat slice;

    int REF_FLAG;
    int GPU_FLAG;
    vector<float> geom_;

};

class Camera {

 public:
    ~Camera() {

    }

    Camera();

    void init(double f, int imsx, int imsy, int gpu);
    void setScene(Scene scene);
    void setLocation(double x, double y, double z);
    void pointAt(double x, double y, double z);
    
    Mat render();
    void renderCPU();
    void renderGPU();

    Mat getP();
    Mat getC();

 private:

    Mat Rt();
    void project();
    double f(double x, double y);
    void img_refrac(Mat_<double> Xcam, Mat_<double> X, Mat_<double> &Xout);

    int imsx_, imsy_, cx_, cy_;
    double f_;

    Mat_<double> C_;
    Mat_<double> t_;
    Mat_<double> R_;
    Mat_<double> K_;
    Mat_<double> P_;

    // Rendered image
    Mat render_;

    // TODO: name these better
    Mat_<double> p_;
    Mat_<double> s_;

    gpu::GpuMat gx, gy, tmp1, tmp2, img;

    Scene scene_;
    int REF_FLAG;
    int GPU_FLAG;
    vector<float> geom_;

};

class benchmark {

 public:
    ~benchmark() {

    }

    benchmark() {}

    void benchmarkSA(Scene scene, saRefocus refocus);
    double calcQ(double thresh, int mult, double mult_exp);

 private:

    Scene scene_;
    saRefocus refocus_;

};

#endif
