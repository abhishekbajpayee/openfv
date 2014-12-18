// -------------------------------------------------------
// -------------------------------------------------------
// Synthetic Aperture - Particle Tracking Velocimetry Code
// --- Rendering Library ---
// -------------------------------------------------------
// Author: Abhishek Bajpayee
//         Dept. of Mechanical Engineering
//         Massachusetts Institute of Technology
// -------------------------------------------------------
// -------------------------------------------------------

#include "std_include.h"
//#include "calibration.h"
//#include "refocusing.h"
#include "tools.h"
//#include "cuda_lib.h"
//#include "visualize.h"
#include "rendering.h"

//#include <Eigen/Core>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace std;
using namespace cv;

void Scene::create(double sx, int vx, double sy, int vy, double sz, int vz) {

    LOG(INFO)<<"Creating scene...";

    xlims_.push_back(-0.5*sx); xlims_.push_back(0.5*sx);
    ylims_.push_back(-0.5*sy); ylims_.push_back(0.5*sy);
    zlims_.push_back(-0.5*sz); zlims_.push_back(0.5*sz);

    sx_ = sx; sy_ = sy; sz_ = sz;
    vx_ = vx; vy_ = vy; vz_ = vz;

    voxelsX_ = linspace(-0.5*sx, 0.5*sx, vx);
    voxelsY_ = linspace(-0.5*sy, 0.5*sy, vy);
    voxelsZ_ = linspace(-0.5*sz, 0.5*sz, vz);

}

void Scene::seedParticles(int num) {

    LOG(INFO)<<"Seeding particles...";

    int res = 1000;

    double x, y, z;
    vector<double> p;
    for (int i=0; i<num; i++) {

        x = (double(rand()%res)/double(res))*sx_ - 0.5*sx_;
        y = (double(rand()%res)/double(res))*sy_ - 0.5*sy_;
        z = (double(rand()%res)/double(res))*sz_ - 0.5*sz_;

        p.push_back(x); p.push_back(y); p.push_back(z);
        particles_.push_back(p);
        p.clear();

    }

    LOG(INFO)<<"Generating voxels...";
    createVolume();

}

void Scene::createVolume() {

    double thresh = 0.1;

    for (int i=0; i<voxelsX_.size(); i++) {
        for (int j=0; j<voxelsY_.size(); j++) {
            for (int k=0; k<voxelsZ_.size(); k++) {

                double intensity = f(voxelsX_[i], voxelsY_[j], voxelsZ_[k]);
                if (intensity > thresh) {
                    voxel v;
                    v.x = i; v.y = j; v.z = k; v.I = int(intensity);
                    volume_.push_back(v);
                }

            }
        }
    }

    VLOG(1)<<"Seeded voxels: "<<volume_.size();

}

double Scene::f(double x, double y, double z) {

    double intensity=0;
    double sigma=1.0;
    double pi=3.14159;

    for (int i=0; i<particles_.size(); i++) {
        intensity += (1.0/2*pi*pow(sigma,2))*exp( -1.0*( pow(x-particles_[i][0], 2), pow(x-particles_[i][1], 2), pow(x-particles_[i][2], 2) )/(2*pow(sigma, 2)) );
    }

    return(intensity*255.0);

}

Mat Scene::getImg(int zv) {

    Mat A = Mat::zeros(vx_, vy_, CV_8U);
    vector<voxel> slice = getVoxels(zv);

    for (int i=0; i<slice.size(); i++) {
        A.at<char>(slice[i].x, slice[i].y) = slice[i].I;
    }
    
    return(A);

}

vector<voxel> Scene::getVoxels(int z) {

    vector<voxel> slice;

    for (int i=0; i<volume_.size(); i++) {
        if (volume_[i].z == z)
            slice.push_back(volume_[i]);
    }
    
    return(slice);

}


