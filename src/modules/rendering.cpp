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
// --- Rendering Library ---
// -------------------------------------------------------
// Author: Abhishek Bajpayee
//         Dept. of Mechanical Engineering
//         Massachusetts Institute of Technology
// -------------------------------------------------------
// -------------------------------------------------------

#include "rendering.h"
#include "tools.h"

using namespace std;
using namespace cv;

// Scene class functions

Scene::Scene() {}

void Scene::create(double sx, double sy, double sz, int gpu) {

    LOG(INFO)<<"Creating scene...";

    xlims_.push_back(-0.5*sx); xlims_.push_back(0.5*sx);
    ylims_.push_back(-0.5*sy); ylims_.push_back(0.5*sy);
    zlims_.push_back(-0.5*sz); zlims_.push_back(0.5*sz);

    sx_ = sx; sy_ = sy; sz_ = sz;

    REF_FLAG = 0;
    geom_.push_back(0); geom_.push_back(0); geom_.push_back(0); geom_.push_back(0); geom_.push_back(0);

    // setting default particle sigma = 0.1 mm
    double sigma = 0.1;
    sigmax_ = sigma;
    sigmay_ = sigma;
    sigmaz_ = sigma;

    frame_ = 0;

    GPU_FLAG = gpu;

#ifdef WITHOUT_CUDA
    if (GPU_FLAG)
        LOG(FATAL)<<"OpenFV was built without CUDA! Switch GPU option to OFF.";
#endif

    CIRC_VOL_FLAG = 0;

}

#ifndef WITHOUT_CUDA
void Scene::setGpuFlag(int gpu) {
    GPU_FLAG = gpu;
}
#endif

void Scene::setCircVolFlag(int flag) {
    CIRC_VOL_FLAG = flag;
}

void Scene::setParticleSigma(double sx, double sy, double sz) {

    LOG(INFO) << "Setting particle sigma to (" << sx << ", " << sy << ", " << sz << ") [mm]"; 

    sigmax_ = sx;
    sigmay_ = sy;
    sigmaz_ = sz;

}

void Scene::setActiveFrame(int frame) {

    if (frame > trajectory_.size()-1)
        LOG(FATAL) << "Active frame number must be <= " << trajectory_.size() << " (size of rendered frames)!";

    frame_ = frame;

}

void Scene::setRefractiveGeom(float zW, float n1, float n2, float n3, float t) {

    REF_FLAG = 1;

    geom_[0] = zW;
    geom_[1] = n1; geom_[2] = n2; geom_[3] = n3;
    geom_[4] = t;

}

void Scene::seedR() {

    LOG(INFO)<<"Seeding R...";

    particles_ = (Mat_<double>(4,13) << 0, 0,   0,   0,   0,   10,  20,  20,  20,  10,  0,   10,  20,
                                        0, -10, -20, -30, -40, -40, -40, -30, -20, -20, -20, -10, 0,
                                        0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                        1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1);

}

void Scene::seedAxes() {

    LOG(INFO)<<"Seeding coordinate system axes...";

    particles_ = Mat_<double>::zeros(4, 175);

    for (int i=0; i<100; i++) {
        particles_(3,i) = 1; particles_(0,i) = i*0.5;
    }
    for (int i=0; i<50; i++) {
        particles_(3,i+100) = 1; particles_(1,i+100) = i;
    }
    for (int i=0; i<25; i++) {
        particles_(3,i+150) = 1; particles_(2,i+150) = i*2;
    }

    // TODO: add axis labels enough to identify orientation

}

void Scene::seedFromFile(string path) {

    ifstream file;
    file.open(path.c_str());

    LOG(INFO)<<"Seeding particles...";

    int num;
    file>>num;
    particles_ = Mat_<double>::zeros(4, num);

    double x, y, z;
    for (int i=0; i<num; i++) {
        file>>x; file>>y; file>>z;
        particles_(0,i) = x; particles_(1,i) = y; particles_(2,i) = z; particles_(3,i) = 1;
    }

    trajectory_.push_back(particles_.clone());

}

void Scene::seedParticles(vector< vector<double> > points) {

    LOG(INFO)<<"Seeding particles...";

    int num = points.size();
    particles_ = Mat_<double>::zeros(4, num);

    for (int i=0; i<num; i++) {
        particles_(0,i) = points[i][0]; particles_(1,i) = points[i][1]; particles_(2,i) = points[i][2]; particles_(3,i) = 1;
    }

    trajectory_.push_back(particles_.clone());

}

void Scene::seedParticles(int num, double factor) {

    LOG(INFO)<<"Seeding particles...";

    particles_ = Mat_<double>::zeros(4, num);

    int res = 1000;

    double x, y, z, r, theta;

    if (CIRC_VOL_FLAG) {

        // double R = sqrt(pow(0.5*sx_,2)+pow(0.5*sy_,2)+pow(0.5*sz_,2))*factor;
        double R = 0.5*sx_*factor;
        LOG(INFO)<<"R: "<<R;
        int i = 0;
        while (i < num) {

            x = (double(rand()%res)/double(res))*factor*sx_ - 0.5*factor*sx_;
            y = (double(rand()%res)/double(res))*factor*sy_ - 0.5*factor*sy_;
            z = (double(rand()%res)/double(res))*factor*sz_ - 0.5*factor*sz_;

            if (sqrt(x*x + y*y + z*z)<R) {
                particles_(0,i) = x; particles_(1,i) = y; particles_(2,i) = z; particles_(3,i) = 1;
                i++;
            }

        }

    } else {

        for (int i=0; i<num; i++) {

            x = (double(rand()%res)/double(res))*factor*sx_ - 0.5*factor*sx_;
            y = (double(rand()%res)/double(res))*factor*sy_ - 0.5*factor*sy_;
            z = (double(rand()%res)/double(res))*factor*sz_ - 0.5*factor*sz_;

            particles_(0,i) = x; particles_(1,i) = y; particles_(2,i) = z; particles_(3,i) = 1;

        }

    }

    trajectory_.push_back(particles_.clone());

}

// Accepts a pointer to a function func(x, y, z, t) which outputs the new position of a particle
// at (x, y, z) after moving it according to a certain velocity field over time t
void Scene::propagateParticles(vector<double> (*func)(double, double, double, double), double t) {

    VLOG(1)<<"Propagating particles using function over "<<t<<" seconds...";

    for (int i=0; i<particles_.cols; i++) {
        vector<double> new_point = func(particles_(0,i), particles_(1,i), particles_(2,i), t);
        particles_(0,i) = new_point[0]; particles_(1,i) = new_point[1]; particles_(2,i) = new_point[2];
    }

    trajectory_.push_back(particles_.clone());

}

void Scene::renderVolume(int xv, int yv, int zv) {

#ifndef WITHOUT_CUDA
    if (GPU_FLAG) {
        renderVolumeGPU2(xv, yv, zv);
    }
#endif
    if (!GPU_FLAG) {
        renderVolumeCPU(xv, yv, zv);
    }

}

void Scene::renderVolumeCPU(int xv, int yv, int zv) {

    volumeCPU_.clear();
    vx_ = xv; vy_ = yv; vz_ = zv;
    voxelsX_ = linspace(-0.5*sx_, 0.5*sx_, vx_);
    voxelsY_ = linspace(-0.5*sy_, 0.5*sy_, vy_);
    voxelsZ_ = linspace(-0.5*sz_, 0.5*sz_, vz_);

    double smax = sigmax_;
    if (sigmay_ > smax)
        smax = sigmay_;
    if (sigmaz_ > smax)
        smax = sigmaz_;

    dthresh_ = -2.0*smax*smax*log(0.001);

    LOG(INFO)<<"CPU rendering voxels...";

    for (int k=0; k<voxelsZ_.size(); k++) {

        LOG(INFO)<<k;

        Mat img = Mat::zeros(vy_, vx_, CV_32F);

        for (int i=0; i<voxelsX_.size(); i++)
            for (int j=0; j<voxelsY_.size(); j++)
                    img.at<float>(j, i) = f(voxelsX_[i], voxelsY_[j], voxelsZ_[k]); //intensity;

        volumeCPU_.push_back(img.clone());

    }

    volumesCPU_.push_back(volumeCPU_);

    VLOG(1)<<"done";

}

void Scene::renderVolumeCPU2(int xv, int yv, int zv) {

    volumeCPU_.clear();
    vx_ = xv; vy_ = yv; vz_ = zv;
    voxelsX_ = linspace(-0.5*sx_, 0.5*sx_, vx_);
    voxelsY_ = linspace(-0.5*sy_, 0.5*sy_, vy_);
    voxelsZ_ = linspace(-0.5*sz_, 0.5*sz_, vz_);

    double smax = sigmax_;
    if (sigmay_ > smax)
        smax = sigmay_;
    if (sigmaz_ > smax)
        smax = sigmaz_;

    dthresh_ = -2.0*smax*smax*log(0.001);

    double ax = (vx_-1)/sx_; double bx = (vx_-1)/2.0;
    double ay = (vy_-1)/sy_; double by = (vy_-1)/2.0;
    double az = (vz_-1)/sz_; double bz = (vz_-1)/2.0;

    if (vx_/sx_ != vy_/sy_ || vx_/sx_ != vz_/sz_)
        LOG(FATAL) << "Different aspect ratios in each direction! This render function probably does not work well for a situation like this.";

    // overestimate voxel threshold
    int vthresh = round(sqrt(dthresh_)*vx_/sx_);

    // init blank volume
    for (int k=0; k<voxelsZ_.size(); k++) {
        Mat img = Mat::zeros(vy_, vx_, CV_32F);
        volumeCPU_.push_back(img.clone());
    }

    int xi, lx, ux, yi, ly, uy, zi, lz, uz;
    double dx, dy, dz, I;
    for (int i=0; i<particles_.cols; i++) {

        LOG(INFO) << i;

        xi = round(particles_(0,i)*ax + bx);
        yi = round(particles_(1,i)*ay + by);
        zi = round(particles_(2,i)*az + bz);
        lx = max(0, xi-vthresh); ux = max(vx_-1, xi+vthresh);
        ly = max(0, yi-vthresh); uy = max(vy_-1, yi+vthresh);
        lz = max(0, zi-vthresh); uz = max(vz_-1, zi+vthresh);

        for (int xii = lx; xii <= ux; xii++) {
            for (int yii = ly; yii <= uy; yii++) {
                for (int zii = lz; zii <= uz; zii++) {

                    dx = pow(voxelsX_[xii]-particles_(0,i), 2);
                    dy = pow(voxelsY_[yii]-particles_(1,i), 2); 
                    dz = pow(voxelsZ_[zii]-particles_(2,i), 2);

                    I = exp( -1.0*(dx/(2*pow(sigmax_, 2)) + dy/(2*pow(sigmay_, 2)) + dz/(2*pow(sigmaz_, 2))) );
                    volumeCPU_[zii].at<float>(yii,xii) += I;

                }
            }
        }

    }

    LOG(FATAL) << "fuck off";

}

double Scene::f(double x, double y, double z) {

    double intensity=0;
    double d, dx, dy, dz, b;

    for (int i=0; i<particles_.cols; i++) {
        dx = pow(x-particles_(0,i), 2); dy = pow(y-particles_(1,i), 2); dz = pow(z-particles_(2,i), 2);
        d = dx + dy + dz;
        if (d < dthresh_) {
            b = exp( -1.0*(dx/(2*pow(sigmax_, 2)) + dy/(2*pow(sigmay_, 2)) + dz/(2*pow(sigmaz_, 2))) );
            intensity += b;
        }
    }

    return(intensity);

}

#ifndef WITHOUT_CUDA

void Scene::renderVolumeGPU(int xv, int yv, int zv) {

    LOG(INFO)<<"GPU rendering voxels...";

    volumeGPU_.clear();
    vx_ = xv; vy_ = yv; vz_ = zv;
    voxelsX_ = linspace(-0.5*sx_, 0.5*sx_, vx_);
    voxelsY_ = linspace(-0.5*sy_, 0.5*sy_, vy_);
    voxelsZ_ = linspace(-0.5*sz_, 0.5*sz_, vz_);

    Mat x = Mat::zeros(vy_, vx_, CV_32F);
    Mat y = Mat::zeros(vy_, vx_, CV_32F);
    Mat blank = Mat::zeros(vy_, vx_, CV_32F);

    // filling temp matrices
    tmp1.upload(blank); tmp2.upload(blank); tmp3.upload(blank); tmp4.upload(blank);
    slice.upload(blank);

    for (int i=0; i<vy_; i++) {
        for (int j=0; j<vx_; j++) {
            x.at<float>(i,j) = voxelsX_[j];
            y.at<float>(i,j) = voxelsY_[i];
        }
    }

    gx.upload(x); gy.upload(y);

    for (int z=0; z<vz_; z++) {

        slice = 0;

        for (int k=0; k<particles_.cols; k++) {

            tmp1 = 0; tmp2 = 0;

            // outputs -(x-ux)^2/2sig^2
            gpu::add(gx, Scalar(-particles_(0,k)), tmp1);
            gpu::pow(tmp1, 2.0, tmp1);
            gpu::multiply(tmp1, Scalar(-1.0/(2*pow(sigmax_, 2))), tmp1);
            gpu::add(tmp1, tmp2, tmp2);

            tmp1 = 0;

            // outputs -(y-uy)^2/2sig^2
            gpu::add(gy, Scalar(-particles_(1,k)), tmp1);
            gpu::pow(tmp1, 2.0, tmp1);
            gpu::multiply(tmp1, Scalar(-1.0/(2*pow(sigmay_, 2))), tmp1);
            gpu::add(tmp1, tmp2, tmp2);

            gpu::add(tmp2, Scalar( -1.0*pow(voxelsZ_[z]-particles_(2,k), 2.0) / (2*pow(sigmaz_, 2)) ), tmp2);

            gpu::exp(tmp2, tmp2);
            gpu::add(slice, tmp2, slice);

        }

        Mat result(slice); 
        volumeGPU_.push_back(result.clone());

    }

    volumesGPU_.push_back(volumeGPU_);

    VLOG(1)<<"done";

}

void Scene::renderVolumeGPU2(int xv, int yv, int zv) {

    LOG(INFO)<<"GPU rendering voxels fast...";

    volumeGPU_.clear();
    vx_ = xv; vy_ = yv; vz_ = zv;
    voxelsX_ = linspace(-0.5*sx_, 0.5*sx_, vx_);
    voxelsY_ = linspace(-0.5*sy_, 0.5*sy_, vy_);
    voxelsZ_ = linspace(-0.5*sz_, 0.5*sz_, vz_);

    Mat x = Mat::zeros(vy_, vx_, CV_32F);
    Mat y = Mat::zeros(vy_, vx_, CV_32F);
    Mat blank = Mat::zeros(vy_, vx_, CV_32F);

    // filling temp matrices
    tmp1.upload(blank); tmp2.upload(blank); tmp3.upload(blank); tmp4.upload(blank);
    slice.upload(blank);

    for (int i=0; i<vy_; i++) {
        for (int j=0; j<vx_; j++) {
            x.at<float>(i,j) = voxelsX_[j];
            y.at<float>(i,j) = voxelsY_[i];
        }
    }

    gx.upload(x); gy.upload(y);

    for (int z=0; z<vz_; z++) {

        slice = 0;

        // sorting particles into bins...
        vector<Mat> partbin;
        for (int j=0; j<particles_.cols; j++) {
            if (abs(particles_(2,j)-voxelsZ_[z])<sigmaz_*5)
                partbin.push_back(particles_.col(j));
        }

        for (int k=0; k<partbin.size(); k++) {

            Mat_<double> particle = partbin[k];

            tmp1 = 0; tmp2 = 0;

            // outputs -(x-ux)^2/2sig^2
            gpu::add(gx, Scalar(-particle(0,0)), tmp1);
            gpu::pow(tmp1, 2.0, tmp1);
            gpu::multiply(tmp1, Scalar(-1.0/(2*pow(sigmax_, 2))), tmp1);
            gpu::add(tmp1, tmp2, tmp2);

            tmp1 = 0;

            // outputs -(y-uy)^2/2sig^2
            gpu::add(gy, Scalar(-particle(1,0)), tmp1);
            gpu::pow(tmp1, 2.0, tmp1);
            gpu::multiply(tmp1, Scalar(-1.0/(2*pow(sigmay_, 2))), tmp1);
            gpu::add(tmp1, tmp2, tmp2);

            gpu::add(tmp2, Scalar( -1.0*pow(voxelsZ_[z]-particle(2,0), 2.0) / (2*pow(sigmaz_, 2)) ), tmp2);

            gpu::exp(tmp2, tmp2);
            gpu::add(slice, tmp2, slice);

        }

        Mat result(slice);
        volumeGPU_.push_back(result.clone());

    }

    volumesGPU_.push_back(volumeGPU_);

    VLOG(1)<<"done";

}

#endif

// Get slice from rendered scene or get equivalent of refocused image
// at given depth
Mat Scene::getSlice(int z_ind) {

    Mat img;

    if (GPU_FLAG) {
        img = volumesGPU_[frame_][z_ind];
    } else {
        img = volumesCPU_[frame_][z_ind];
    }

    return(img);

}

vector<Mat> Scene::getVolume() {

    if (GPU_FLAG) {
        return(volumesGPU_[frame_]);
    } else {
        return(volumesCPU_[frame_]);
    }

}

Mat Scene::getParticles() {

    VLOG(2) << "Retreiving particles for frame " << frame_;
    return(trajectory_[frame_]);

}

vector<float> Scene::getRefGeom() {
    return(geom_);
}

int Scene::getRefFlag() {
    return(REF_FLAG);
}

vector<int> Scene::getVoxelGeom() {

    vector<int> geom;
    geom.push_back(vx_);
    geom.push_back(vy_);
    geom.push_back(vz_);

    return(geom);

}

vector<double> Scene::getSceneGeom() {

    vector<double> geom;
    geom.push_back(sx_);
    geom.push_back(sy_);
    geom.push_back(sz_);

    return(geom);

}

double Scene::sigma() {
    return(sigmax_);
}

void Scene::temp() {

    LOG(INFO)<<"State";

    LOG(INFO)<<sigmax_;
    LOG(INFO)<<sigmay_;
    LOG(INFO)<<sigmaz_;

    // LOG(INFO)<<sx_;
    // LOG(INFO)<<sy_;
    // LOG(INFO)<<sz_;

    // LOG(INFO)<<vx_;
    // LOG(INFO)<<vy_;
    // LOG(INFO)<<vz_;

    // LOG(INFO)<<xlims_.size();
    // LOG(INFO)<<voxelsX_.size();

    // LOG(INFO)<<particles_.cols;
    // LOG(INFO)<<particles_;

    LOG(INFO)<<volumeGPU_.size();
    LOG(INFO)<<volumeCPU_.size();

    LOG(INFO)<<trajectory_.size();

    LOG(INFO)<<GPU_FLAG;
    LOG(INFO)<<REF_FLAG;

}

void Scene::dumpStack(string path) {

    imageIO io(path);

#ifndef WITHOUT_CUDA
    if (GPU_FLAG) {
        io<<volumesGPU_[frame_];
    }
#endif
    if (!GPU_FLAG) {
        io<<volumesCPU_[frame_];
    }

}

// Camera class functions

Camera::Camera() {

    K_ = Mat_<double>::zeros(3,3);
    R_ = Mat_<double>::zeros(3,3);
    C_ = Mat_<double>::zeros(3,1);
    t_ = Mat_<double>::zeros(3,1);

}

void Camera::init(double f, int imsx, int imsy, int gpu) {

    VLOG(1)<<"Initializing camera...";

    f_ = f;
    imsx_ = imsx;
    imsy_ = imsy;
    cx_ = 0.5*imsx;
    cy_ = 0.5*imsy;

    // K matrix
    K_(0,0) = f_;  K_(1,1) = f_;
    K_(0,2) = cx_; K_(1,2) = cy_;
    K_(2,2) = 1;

    GPU_FLAG = gpu;

#ifdef WITHOUT_CUDA
    if (GPU_FLAG)
        LOG(FATAL)<<"OpenFV was compiled without CUDA! Switch GPU option to OFF.";
#endif

    CUSTOM_PARTICLE_SIGMA = 0;

}

void Camera::setLocation(double x, double y, double z) {

    C_(0,0) = x; C_(1,0) = y; C_(2,0) = z;
    if (REF_FLAG)
        pointAt(-1*ref_shift_*x, -1*ref_shift_*y, 0);
    else
        pointAt(0, 0, 0);

}

// Generates rotation matrix according to the right, up, out
// convention and point the camera at C to point p
void Camera::pointAt(double x, double y, double z) {

    Mat_<double> p = (Mat_<double>(3,1) << x, y, z);
    Mat_<double> up = (Mat_<double>(3,1) << 0, 1, 0);

    Mat_<double> L = p - C_; L = normalize(L);
    Mat_<double> s = cross(L, up);
    Mat_<double> u = cross(s, L);

    for (int i=0; i<3; i++) {
        R_(0,i) = -s(i,0);
        R_(1,i) = u(i,0);
        R_(2,i) = -L(i,0);
    }

    t_ = -R_*C_;

    P_ = K_*Rt();

}

void Camera::setCustomParticleSigma(double sigma) {

    CUSTOM_PARTICLE_SIGMA = 1;
    custom_sigma_ = sigma;

}

void Camera::setScene(Scene &scene) {

    scene_ = scene;
    REF_FLAG = scene_.getRefFlag();
    if (REF_FLAG)
        geom_ = scene.getRefGeom();

}

Mat Camera::render() {

#ifndef WITHOUT_CUDA
    if (GPU_FLAG) {
        renderGPU();
    }
#endif

    if (!GPU_FLAG) {
        renderCPU();
    }

    return(render_);

}

void Camera::renderCPU() {

    project();

    VLOG(1)<<"Rendering image...";

    Mat img = Mat::zeros(imsy_, imsx_, CV_32F);

    for (int i=0; i<img.rows; i++) {
        for (int j=0; j<img.cols; j++) {
            img.at<float>(i,j) = f(double(j), double(i));
        }
    }

    render_ = img.clone();

}

double Camera::f(double x, double y) {

    double intensity=0;

    for (int i=0; i<p_.cols; i++) {
        double d = pow(x-p_(0,i), 2) + pow(y-p_(1,i), 2);
        if (d<25)
        {
            double b = exp( -d/(2*pow(s_(0,i), 2)) );
            intensity += b;
        }

    }

    return(intensity);

}

#ifndef WITHOUT_CUDA
void Camera::renderGPU() {

    project();

    VLOG(1)<<"Rendering image...";

    Mat x = Mat::zeros(imsy_, imsx_, CV_32F);
    Mat y = Mat::zeros(imsy_, imsx_, CV_32F);
    Mat blank = Mat::zeros(imsy_, imsx_, CV_32F);

    tmp1.upload(blank), tmp2.upload(blank);
    img.upload(blank);

    for (int i=0; i<imsy_; i++) {
        for (int j=0; j<imsx_; j++) {
            x.at<float>(i,j) = float(j);
            y.at<float>(i,j) = float(i);
        }
    }

    gx.upload(x); gy.upload(y);

    for (int k=0; k<p_.cols; k++) {

        tmp1 = 0; tmp2 = 0;

        // outputs -(x-ux)^2/2sig^2
        gpu::add(gx, Scalar(-p_(0,k)), tmp1);
        gpu::pow(tmp1, 2.0, tmp1);
        gpu::multiply(tmp1, Scalar(-1.0/(2*s_(0,k)*s_(0,k))), tmp1);
        gpu::add(tmp2, tmp1, tmp2);

        tmp1 = 0;

        // outputs -(y-uy)^2/2sig^2
        gpu::add(gy, Scalar(-p_(1,k)), tmp1);
        gpu::pow(tmp1, 2.0, tmp1);
        gpu::multiply(tmp1, Scalar(-1.0/(2*s_(0,k)*s_(0,k))), tmp1);
        gpu::add(tmp2, tmp1, tmp2);

        gpu::exp(tmp2, tmp2);
        gpu::add(img, tmp2, img);

    }

    Mat result(img);
    render_ = result.clone();

}
#endif

Mat Camera::getP() {
    return(P_.clone());
}

Mat Camera::getC() {
    return(C_.clone());
}

// TODO: Consider GPU version of this?
void Camera::project() {

    VLOG(1)<<"Projecting points...";

    // project the points + sigmas
    Mat_<double> particles = scene_.getParticles();

    p_ = Mat_<double>::zeros(2, particles.cols);
    s_ = Mat_<double>::zeros(1, particles.cols);

    Mat_<double> proj;
    if (REF_FLAG) {
        Mat_<double> Xout = Mat_<double>::zeros(4, particles.cols);
        img_refrac(C_, particles, Xout);
        proj = P_*Xout;
    } else {
        proj = P_*particles;
    }
    double d;

    // TODO: optimize, consider wrapping in Eigen Matrix containers
    for (int i=0; i<proj.cols; i++) {

        proj(0,i) /= proj(2,i);
        proj(1,i) /= proj(2,i);
        p_(0,i) = proj(0,i);
        p_(1,i) = proj(1,i);

        // TODO: correct this for refractive and decide if needed at all because of diffraction limited imaging
        if (CUSTOM_PARTICLE_SIGMA) {
            s_(0,i) = custom_sigma_;
        } else {
            d = sqrt( pow(C_(0,0)-particles(0,i), 2) + pow(C_(1,0)-particles(1,i), 2) + pow(C_(2,0)-particles(2,i), 2) );
            // converting scene particle sigma from mm to pixels
            s_(0,i) = scene_.sigma()*f_/d;
        }
        VLOG(3)<<"Particle sigma: "<<s_(0,i);

    }

}

void Camera::img_refrac(Mat_<double> Xcam, Mat_<double> X, Mat_<double> &X_out) {

    float zW_ = geom_[0]; float n1_ = geom_[1]; float n2_ = geom_[2]; float n3_ = geom_[3]; float t_ = geom_[4];

    double c[3];
    for (int i=0; i<3; i++)
        c[i] = Xcam.at<double>(i,0);

    for (int n=0; n<X.cols; n++) {

        double a[3];
        double b[3];
        double point[3];
        for (int i=0; i<3; i++)
            point[i] = X(i,n);

        a[0] = c[0] + (point[0]-c[0])*(zW_-c[2])/(point[2]-c[2]);
        a[1] = c[1] + (point[1]-c[1])*(zW_-c[2])/(point[2]-c[2]);
        a[2] = zW_;
        b[0] = c[0] + (point[0]-c[0])*(t_+zW_-c[2])/(point[2]-c[2]);
        b[1] = c[1] + (point[1]-c[1])*(t_+zW_-c[2])/(point[2]-c[2]);
        b[2] = t_+zW_;

        double rp = sqrt( pow(point[0]-c[0],2) + pow(point[1]-c[1],2) );
        double dp = point[2]-b[2];
        double phi = atan2(point[1]-c[1],point[0]-c[0]);

        double ra = sqrt( pow(a[0]-c[0],2) + pow(a[1]-c[1],2) );
        double rb = sqrt( pow(b[0]-c[0],2) + pow(b[1]-c[1],2) );
        double da = a[2]-c[2];
        double db = b[2]-a[2];

        double f, g, dfdra, dfdrb, dgdra, dgdrb;

        // Newton Raphson loop to solve for Snell's law
        double tol=1E-8;

        for (int i=0; i<10; i++) {

            f = ( ra/sqrt(pow(ra,2)+pow(da,2)) ) - ( (n2_/n1_)*(rb-ra)/sqrt(pow(rb-ra,2)+pow(db,2)) );
            g = ( (rb-ra)/sqrt(pow(rb-ra,2)+pow(db,2)) ) - ( (n3_/n2_)*(rp-rb)/sqrt(pow(rp-rb,2)+pow(dp,2)) );

            dfdra = ( (1.0)/sqrt(pow(ra,2)+pow(da,2)) )
                - ( pow(ra,2)/pow(pow(ra,2)+pow(da,2),1.5) )
                + ( (n2_/n1_)/sqrt(pow(ra-rb,2)+pow(db,2)) )
                - ( (n2_/n1_)*(ra-rb)*(2*ra-2*rb)/(2*pow(pow(ra-rb,2)+pow(db,2),1.5)) );

            dfdrb = ( (n2_/n1_)*(ra-rb)*(2*ra-2*rb)/(2*pow(pow(ra-rb,2)+pow(db,2),1.5)) )
                - ( (n2_/n1_)/sqrt(pow(ra-rb,2)+pow(db,2)) );

            dgdra = ( (ra-rb)*(2*ra-2*rb)/(2*pow(pow(ra-rb,2)+pow(db,2),1.5)) )
                - ( (1.0)/sqrt(pow(ra-rb,2)+pow(db,2)) );

            dgdrb = ( (1.0)/sqrt(pow(ra-rb,2)+pow(db,2)) )
                + ( (n3_/n2_)/sqrt(pow(rb-rp,2)+pow(dp,2)) )
                - ( (ra-rb)*(2*ra-2*rb)/(2*pow(pow(ra-rb,2)+pow(db,2),1.5)) )
                - ( (n3_/n2_)*(rb-rp)*(2*rb-2*rp)/(2*pow(pow(rb-rp,2)+pow(dp,2),1.5)) );

            ra = ra - ( (f*dgdrb - g*dfdrb)/(dfdra*dgdrb - dfdrb*dgdra) );
            rb = rb - ( (g*dfdra - f*dgdra)/(dfdra*dgdrb - dfdrb*dgdra) );

        }

        a[0] = ra*cos(phi) + c[0];
        a[1] = ra*sin(phi) + c[1];

        X_out(0,n) = a[0];
        X_out(1,n) = a[1];
        X_out(2,n) = a[2];
        X_out(3,n) = 1.0;

    }

}

Mat Camera::Rt() {

    Mat_<double> Rt = Mat_<double>::zeros(3,4);

    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            Rt(i,j) = R_(i,j);
        }
        Rt(i,3) = t_(i,0);
    }

    return(Rt);

}

/*
void benchmark::benchmarkSA(Scene scn, saRefocus refocus) {

    scene_ = scn;
    refocus_ = refocus;

}

double benchmark::calcQ(double thresh, int mult, double mult_exp) {

    vector<int> voxels = scene_.getVoxelGeom();
    vector<double> scnSize = scene_.getSceneGeom();
    vector<double> z = linspace(-0.5*scnSize[2], 0.5*scnSize[2], voxels[2]);

    if (mult) {
        refocus_.setMult(mult, mult_exp);
    } else {
        refocus_.setMult(0, 1.0);
    }

    double at = 0; double bt = 0; double ct = 0;
    for (int i=0; i<voxels[2]; i++) {

        Mat ref = scene_.getSlice(i);

        Mat img;
        if (mult) {
            img = refocus_.refocus(z[i], 0, 0, 0, 0, 0); // <-- TODO: in future add ability to handle multiple time frames?
        } else {
            img = refocus_.refocus(z[i], 0, 0, 0, thresh, 0);
        }

        //qimshow(ref); qimshow(img);

        //double minval, maxval; minMaxLoc(img, minval, maxval);
        //VLOG(3)<<maxval;

        Mat a; multiply(ref, img, a); double as = double(sum(a)[0]); at += as;
        Mat b; pow(ref, 2, b); double bs = double(sum(b)[0]); bt += bs;
        Mat c; pow(img, 2, c); double cs = double(sum(c)[0]); ct += cs;

    }

    double Q = at/sqrt(bt*ct);
    VLOG(1)<<"Q data: "<<at<<", "<<bt<<", "<<ct<<", "<<Q;

    return(Q);

}
*/

// Python wrapper
BOOST_PYTHON_MODULE(rendering) {

    using namespace boost::python;

    void (Scene::*sPx1)(vector< vector<double> >) = &Scene::seedParticles;
    void (Scene::*sPx2)(int, double)              = &Scene::seedParticles;

    class_<Scene>("Scene")
        .def("create", &Scene::create)
#ifndef WITHOUT_CUDA
        .def("setGpuFlag", &Scene::setGpuFlag)
#endif
        .def("seedR", &Scene::seedR)
        .def("seedParticles", sPx2)
        .def("setParticleSigma", &Scene::setParticleSigma)
        .def("setRefractiveGeom", &Scene::setRefractiveGeom)
        .def("renderVolume", &Scene::renderVolume)
        .def("getSlice", &Scene::getSlice)
    ;

    class_<Camera>("Camera")
        .def("init", &Camera::init)
        .def("setScene", &Camera::setScene)
        .def("setLocation", &Camera::setLocation)
        .def("render", &Camera::render)
        .def("getP", &Camera::getP)
        .def("getC", &Camera::getC)
    ;

    /*
    class_<benchmark>("benchmark")
        .def("benchmarkSA", &benchmark::benchmarkSA)
        .def("calcQ", &benchmark::calcQ)
    ;
    */

}
