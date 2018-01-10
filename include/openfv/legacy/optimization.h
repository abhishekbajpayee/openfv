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

#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include "std_include.h"

using namespace cv;
using namespace std;

// CLASS AND STRUCT DEFINITIONS

// Read a Bundle Adjustment dataset

// Container class for a pinhole bundle adjustment dataset
class baProblem {

 public:
    ~baProblem() {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] plane_index_;
        delete[] observations_;
        delete[] parameters_;
    }

    baProblem() {
        point_index_ = NULL;
        camera_index_ = NULL;
        plane_index_ = NULL;
        observations_ = NULL;
        parameters_ = NULL;
    }

    int num_observations()       const { return num_observations_;               }
    const double* observations() const { return observations_;                   }
    double* mutable_cameras()          { return parameters_;                     }
    double* mutable_points()           { return parameters_ + 9 * num_cameras_; }
    double* mutable_planes()           { return parameters_ + 9 * num_cameras_ + 3 * num_points_; }

    int num_cameras()                  { return num_cameras_;                    }
    int num_points()                   { return num_points_;                     }
    int num_planes()                   { return num_planes_;                     }
    int* camera_index()                { return camera_index_;                   }
    int* point_index()                 { return point_index_;                    }

    double* mutable_camera_for_observation(int i) {
        return mutable_cameras() + camera_index_[i] * 9;
    }
    double* mutable_point_for_observation(int i) {
        return mutable_points() + point_index_[i] * 3;
    }
    double* mutable_plane_for_observation(int i) {
        return mutable_planes() + plane_index_[i] * 4;
    }

    bool LoadFile(const char* filename) {
        FILE* fptr = fopen(filename, "r");
        if (fptr == NULL) {
            return false;
        };

        FscanfOrDie(fptr, "%d", &num_cameras_);
        FscanfOrDie(fptr, "%d", &num_planes_);
        FscanfOrDie(fptr, "%d", &num_points_);
        FscanfOrDie(fptr, "%d", &num_observations_);

        point_index_ = new int[num_observations_];
        camera_index_ = new int[num_observations_];
        plane_index_ = new int[num_observations_];
        observations_ = new double[2 * num_observations_];

        num_parameters_ = (9 * num_cameras_) + (3 * num_points_) + (4 * num_planes_);
        parameters_ = new double[num_parameters_];

        for (int i = 0; i < num_observations_; ++i) {
            FscanfOrDie(fptr, "%d", camera_index_ + i);
            FscanfOrDie(fptr, "%d", plane_index_ + i);
            FscanfOrDie(fptr, "%d", point_index_ + i);
            for (int j = 0; j < 2; ++j) {
                FscanfOrDie(fptr, "%lf", observations_ + 2*i + j);
            }
        }

        for (int i = 0; i < num_parameters_; ++i) {
            FscanfOrDie(fptr, "%lf", parameters_ + i);

        }
        return true;
    }

    double cx;
    double cy;
    double scale;

 private:
    template<typename T>
        void FscanfOrDie(FILE *fptr, const char *format, T *value) {
        int num_scanned = fscanf(fptr, format, value);
        if (num_scanned != 1) {
            LOG(FATAL) << "Invalid UW data file.";
        }
    }

    int num_cameras_;
    int num_planes_;
    int num_points_;
    int* point_index_;
    int* camera_index_;
    int* plane_index_;
    int num_observations_;
    int num_parameters_;

    double* observations_;
    double* parameters_;
};

// Container class for a refractive bundle adjustment dataset
class baProblem_ref {

 public:
    ~baProblem_ref() {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] plane_index_;
        delete[] observations_;
        delete[] parameters_;
    }

    baProblem_ref() {
        point_index_ = NULL;
        camera_index_ = NULL;
        plane_index_ = NULL;
        observations_ = NULL;
        parameters_ = NULL;
    }

    int num_observations()       const { return num_observations_;               }
    const double* observations() const { return observations_;                   }
    double* mutable_cameras()          { return parameters_;                     }
    double* mutable_points()           { return parameters_ + 9 * num_cameras_; }
    double* mutable_planes()           { return parameters_ + 9 * num_cameras_ + 3 * num_points_; }

    int num_cameras()                  { return num_cameras_;                    }
    int num_points()                   { return num_points_;                     }
    int num_planes()                   { return num_planes_;                     }
    int* camera_index()                { return camera_index_;                   }
    int* point_index()                 { return point_index_;                    }

    double t()                         { return t_;  }
    double n1()                        { return n1_; }
    double n2()                        { return n2_; }
    double n3()                        { return n3_; }
    double z0()                        { return z0_; }

    double* mutable_camera_for_observation(int i) {
        return mutable_cameras() + camera_index_[i] * 9;
    }
    double* mutable_point_for_observation(int i) {
        return mutable_points() + point_index_[i] * 3;
    }
    double* mutable_plane_for_observation(int i) {
        return mutable_planes() + plane_index_[i] * 4;
    }

    bool LoadFile(const char* filename) {
        FILE* fptr = fopen(filename, "r");
        if (fptr == NULL) {
            return false;
        };

        FscanfOrDie(fptr, "%d", &num_cameras_);
        FscanfOrDie(fptr, "%d", &num_planes_);
        FscanfOrDie(fptr, "%d", &num_points_);
        FscanfOrDie(fptr, "%d", &num_observations_);

        point_index_ = new int[num_observations_];
        camera_index_ = new int[num_observations_];
        plane_index_ = new int[num_observations_];
        observations_ = new double[2 * num_observations_];

        num_parameters_ = (9 * num_cameras_) + (3 * num_points_) + (4 * num_planes_);
        parameters_ = new double[num_parameters_];

        for (int i = 0; i < num_observations_; ++i) {
            FscanfOrDie(fptr, "%d", camera_index_ + i);
            FscanfOrDie(fptr, "%d", plane_index_ + i);
            FscanfOrDie(fptr, "%d", point_index_ + i);
            for (int j = 0; j < 2; ++j) {
                FscanfOrDie(fptr, "%lf", observations_ + 2*i + j);
            }
        }

        for (int i = 0; i < num_parameters_; ++i) {
            FscanfOrDie(fptr, "%lf", parameters_ + i);
        }

        FscanfOrDie(fptr, "%lf", &t_);
        FscanfOrDie(fptr, "%lf", &n1_);
        FscanfOrDie(fptr, "%lf", &n2_);
        FscanfOrDie(fptr, "%lf", &n3_);
        FscanfOrDie(fptr, "%lf", &z0_);

        return true;
    }

    double cx;
    double cy;
    double scale;

 private:
    template<typename T>
        void FscanfOrDie(FILE *fptr, const char *format, T *value) {
        int num_scanned = fscanf(fptr, format, value);
        if (num_scanned != 1) {
            LOG(FATAL) << "Invalid UW data file.";
        }
    }

    int num_cameras_;
    int num_planes_;
    int num_points_;
    int* point_index_;
    int* camera_index_;
    int* plane_index_;
    int num_observations_;
    int num_parameters_;

    double* observations_;
    double* parameters_;

    double t_, n1_, n2_, n3_, z0_;

};

class baProblem_plane {

 public:
    ~baProblem_plane() {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] plane_index_;
        delete[] observations_;
        delete[] parameters_;
    }

    baProblem_plane() {
        point_index_ = NULL;
        camera_index_ = NULL;
        plane_index_ = NULL;
        observations_ = NULL;
        parameters_ = NULL;
    }

    int num_observations()       const { return num_observations_;               }
    const double* observations() const { return observations_;                   }
    double* mutable_cameras()          { return parameters_;                     }
    double* mutable_planes()           { return parameters_ + 9 * num_cameras_; }

    int num_cameras()                  { return num_cameras_;                    }
    int num_planes()                   { return num_planes_;                     }
    int num_points()                   { return num_points_;                     }
    int* camera_index()                { return camera_index_;                   }
    int* point_index()                 { return point_index_;                    }
    int* plane_index()                 { return plane_index_;                    }

    double t()                         { return t_;  }
    double n1()                        { return n1_; }
    double n2()                        { return n2_; }
    double n3()                        { return n3_; }
    double z0()                        { return z0_; }

    double* mutable_camera_for_observation(int i) {
        return mutable_cameras() + camera_index_[i] * 9;
    }
    double* mutable_plane_for_observation(int i) {
        return mutable_planes() + plane_index_[i] * 6;
    }

    bool LoadFile(const char* filename) {
        FILE* fptr = fopen(filename, "r");
        if (fptr == NULL) {
            return false;
        };

        FscanfOrDie(fptr, "%d", &num_cameras_);
        FscanfOrDie(fptr, "%d", &num_planes_);
        FscanfOrDie(fptr, "%d", &num_points_);
        FscanfOrDie(fptr, "%d", &num_observations_);

        camera_index_ = new int[num_observations_];
        plane_index_ = new int[num_observations_];
        point_index_ = new int[num_observations_];
        observations_ = new double[2 * num_observations_];

        num_parameters_ = (9 * num_cameras_) + (6 * num_planes_);
        parameters_ = new double[num_parameters_];

        for (int i = 0; i < num_observations_; ++i) {
            FscanfOrDie(fptr, "%d", camera_index_ + i);
            FscanfOrDie(fptr, "%d", plane_index_ + i);
            FscanfOrDie(fptr, "%d", point_index_ + i);
            for (int j = 0; j < 2; ++j) {
                FscanfOrDie(fptr, "%lf", observations_ + 2*i + j);
            }
        }

        for (int i = 0; i < num_parameters_; ++i) {
            FscanfOrDie(fptr, "%lf", parameters_ + i);
        }

        FscanfOrDie(fptr, "%lf", &t_);
        FscanfOrDie(fptr, "%lf", &n1_);
        FscanfOrDie(fptr, "%lf", &n2_);
        FscanfOrDie(fptr, "%lf", &n3_);
        FscanfOrDie(fptr, "%lf", &z0_);

        return true;
    }

    double cx;
    double cy;
    double scale;

 private:
    template<typename T>
        void FscanfOrDie(FILE *fptr, const char *format, T *value) {
        int num_scanned = fscanf(fptr, format, value);
        if (num_scanned != 1) {
            LOG(FATAL) << "Invalid UW data file.";
        }
    }

    int num_cameras_;
    int num_planes_;
    int num_points_;
    int* point_index_;
    int* camera_index_;
    int* plane_index_;
    int num_observations_;
    int num_parameters_;

    double* observations_;
    double* parameters_;

    double t_, n1_, n2_, n3_, z0_;

};

// Pinhole Reprojection Error function
class pinholeReprojectionError {

 public:

 pinholeReprojectionError(double observed_x, double observed_y, double cx, double cy, int num_cams):
    observed_x(observed_x), observed_y(observed_y), cx(cx), cy(cy), num_cams(num_cams) {}

    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const {

        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);

        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // Compute the center of distortion.
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        // Image principal points
        T px = T(cx);
        T py = T(cy);

        // Apply second and fourth order radial distortion.
        const T& l1 = camera[7];
        const T& l2 = camera[8];
        // T r2 = xp*xp + yp*yp;
        // T distortion = T(1.0) + r2  * (l1 + l2  * r2);

        // Compute final projected point position.
        const T& focal = camera[6];
        //T predicted_x = (focal * distortion * xp) + px;
        //T predicted_y = (focal * distortion * yp) + py;
        T predicted_x = (focal * xp) + px;
        T predicted_y = (focal * yp) + py;

        // The error is the squared euclidian distance between the predicted and observed position.
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);

        return true;
    }

    double observed_x;
    double observed_y;
    double cx;
    double cy;
    int num_cams;

};

// Pinhole Reprojection Error function with radial distortion
class pinholeReprojectionError_dist {

 public:

 pinholeReprojectionError_dist(double observed_x, double observed_y, double cx, double cy, int num_cams):
    observed_x(observed_x), observed_y(observed_y), cx(cx), cy(cy), num_cams(num_cams) {}

    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const {

        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);

        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // Compute the center of distortion.
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        // Image principal points
        T px = T(cx);
        T py = T(cy);

        // Apply second and fourth order radial distortion.
        const T& l1 = camera[7];
        const T& l2 = camera[8];
        T r2 = xp*xp + yp*yp;
        T distortion = T(1.0) + r2  * (l1 + l2  * r2);

        // Compute final projected point position.
        const T& focal = camera[6];
        //T predicted_x = (focal * distortion * xp) + px;
        //T predicted_y = (focal * distortion * yp) + py;
        T predicted_x = (focal * xp * distortion) + px;
        T predicted_y = (focal * yp * distortion) + py;

        // The error is the squared euclidian distance between the predicted and observed position.
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);

        return true;
    }

    double observed_x;
    double observed_y;
    double cx;
    double cy;
    int num_cams;

};

// Error of points from a plane in space
class planeError {

 public:

 planeError(int num_cams): num_cams(num_cams) {}

    template <typename T>
        bool operator()(const T* const point,
                        const T* const plane,
                        T* residuals) const {

        residuals[0] = plane[0]*point[0] + plane[1]*point[1] + plane[2]*point[2] + plane[3];
        residuals[0] /= sqrt( pow(plane[0],2) + pow(plane[1],2) + pow(plane[2],2) );
        residuals[0] /= T(num_cams);

        return true;

    }

 private:

    int num_cams;

};

// Grid physical size constraint
class gridPhysSizeError {

 public:

 gridPhysSizeError(double grid_phys_size, int gridx, int gridy):
    grid_phys_size(grid_phys_size), gridx(gridx), gridy(gridy) {}

    template <typename T>
        bool operator()(const T* const p1,
                        const T* const p2,
                        const T* const p3,
                        T* residuals) const {

        residuals[0] = sqrt(pow(p1[0]-p2[0],2) + pow(p1[1]-p2[1],2) + pow(p1[2]-p2[2],2))
            - T((gridx-1)*grid_phys_size);
        residuals[0] += sqrt(pow(p1[0]-p3[0],2) + pow(p1[1]-p3[1],2) + pow(p1[2]-p3[2],2))
            - T((gridy-1)*grid_phys_size);

        return true;
    }

 private:

    double grid_phys_size;
    int gridx;
    int gridy;

};

class zError {

 public:

 zError() {}

    template <typename T>
        bool operator()(const T* const p1,
                        const T* const p2,
                        const T* const p3,
                        const T* const p4,
                        T* residuals) const {

        residuals[0] = pow(p1[2]-T(0),2)+pow(p2[2]-T(0),2)+pow(p3[2]-T(0),2)+pow(p4[2]-T(0),2);

        return true;
    }

};

class zError2 {

 public:

 zError2() {}

    template <typename T>
        bool operator()(const T* const p,
                        T* residuals) const {

        residuals[0] = pow(p[0]-T(0),2)+pow(p[1]-T(0),2)+pow(p[2]-T(1),2)+pow(p[4]-T(0),2);

        return true;
    }

};

// xy plane constraint
class xyPlaneError {

 public:

 xyPlaneError() {}

    template <typename T>
        bool operator()(const T* const plane,
                        T* residuals) const {

        residuals[0] = pow(plane[0]-T(0),2)+pow(plane[1]-T(0),2)+pow(plane[2]-T(1),2)+pow(plane[3]-T(0),2);

        return true;
    }

};

// Refractive Reprojection Error function
class refractiveReprojectionError {

 public:

 refractiveReprojectionError(double observed_x, double observed_y, double cx, double cy, int num_cams, double t, double n1, double n2, double n3, double z0)
     : observed_x(observed_x), observed_y(observed_y), cx(cx), cy(cy), num_cams(num_cams), t_(t), n1_(n1), n2_(n2), n3_(n3), z0_(z0) { }

    template <typename T>
        bool operator()(const T* const camera,
                        const T* const point,
                        T* residuals) const {

        // Inital guess for points on glass
        T* R = new T[9];
        ceres::AngleAxisToRotationMatrix(camera, R);

        T c[3];
        for (int i=0; i<3; i++) {
            c[i] = T(0);
            for (int j=0; j<3; j++) {
                c[i] += -R[i*1 + j*3]*camera[j+3];
            }
        }

        // All the refraction stuff
        T* a = new T[3];
        T* b = new T[3];
        a[0] = c[0] + (point[0]-c[0])*(T(-t_)+T(z0_)-c[2])/(point[2]-c[2]);
        a[1] = c[1] + (point[1]-c[1])*(T(-t_)+T(z0_)-c[2])/(point[2]-c[2]);
        a[2] = T(-t_)+T(z0_);
        b[0] = c[0] + (point[0]-c[0])*(T(z0_)-c[2])/(point[2]-c[2]);
        b[1] = c[1] + (point[1]-c[1])*(T(z0_)-c[2])/(point[2]-c[2]);
        b[2] = T(z0_);

        T rp = sqrt( pow(point[0]-c[0],2) + pow(point[1]-c[1],2) );
        T dp = point[2]-b[2];
        T phi = atan2(point[1]-c[1],point[0]-c[0]);

        T ra = sqrt( pow(a[0]-c[0],2) + pow(a[1]-c[1],2) );
        T rb = sqrt( pow(b[0]-c[0],2) + pow(b[1]-c[1],2) );
        T da = a[2]-c[2];
        T db = b[2]-a[2];

        T f, g, dfdra, dfdrb, dgdra, dgdrb;

        // Newton Raphson loop to solve for Snell's law
        for (int i=0; i<20; i++) {

            f = ( ra/sqrt(pow(ra,2)+pow(da,2)) ) - ( T(n2_/n1_)*(rb-ra)/sqrt(pow(rb-ra,2)+pow(db,2)) );
            g = ( (rb-ra)/sqrt(pow(rb-ra,2)+pow(db,2)) ) - ( T(n3_/n2_)*(rp-rb)/sqrt(pow(rp-rb,2)+pow(dp,2)) );

            dfdra = ( T(1.0)/sqrt(pow(ra,2)+pow(da,2)) )
                - ( pow(ra,2)/pow(pow(ra,2)+pow(da,2),1.5) )
                + ( T(n2_/n1_)/sqrt(pow(ra-rb,2)+pow(db,2)) )
                - ( T(n2_/n1_)*(ra-rb)*(T(2)*ra-T(2)*rb)/(T(2)*pow(pow(ra-rb,2)+pow(db,2),1.5)) );

            dfdrb = ( T(n2_/n1_)*(ra-rb)*(T(2)*ra-T(2)*rb)/(T(2)*pow(pow(ra-rb,2)+pow(db,2),1.5)) )
                - ( T(n2_/n1_)/sqrt(pow(ra-rb,2)+pow(db,2)) );

            dgdra = ( (ra-rb)*(T(2)*ra-T(2)*rb)/(T(2)*pow(pow(ra-rb,2)+pow(db,2),1.5)) )
                - ( T(1.0)/sqrt(pow(ra-rb,2)+pow(db,2)) );

            dgdrb = ( T(1.0)/sqrt(pow(ra-rb,2)+pow(db,2)) )
                + ( T(n3_/n2_)/sqrt(pow(rb-rp,2)+pow(dp,2)) )
                - ( (ra-rb)*(T(2)*ra-T(2)*rb)/(T(2)*pow(pow(ra-rb,2)+pow(db,2),1.5)) )
                - ( T(n3_/n2_)*(rb-rp)*(T(2)*rb-T(2)*rp)/(T(2)*pow(pow(rb-rp,2)+pow(dp,2),1.5)) );

            ra = ra - ( (f*dgdrb - g*dfdrb)/(dfdra*dgdrb - dfdrb*dgdra) );
            rb = rb - ( (g*dfdra - f*dgdra)/(dfdra*dgdrb - dfdrb*dgdra) );

        }

        a[0] = ra*cos(phi) + c[0];
        a[1] = ra*sin(phi) + c[1];

        // Continuing projecting point a to camera
        T p[3];
        ceres::AngleAxisRotatePoint(camera, a, p);

        // camera[3,4,5] are the translation.
        p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];

        // Compute the center of distortion.
        T xp = p[0] / p[2]; T yp = p[1] / p[2];

        // Image principal points
        T px = T(cx); T py = T(cy);

        // Apply second and fourth order radial distortion.
        const T& l1 = camera[7];
        const T& l2 = camera[8];
        T r2 = xp*xp + yp*yp;
        T distortion = T(1.0) + r2  * (l1 + l2  * r2);

        // Compute final projected point position.
        const T& focal = camera[6];
        //T predicted_x = (focal * distortion * xp) + px;
        //T predicted_y = (focal * distortion * yp) + py;
        T predicted_x = (focal * xp) + px;
        T predicted_y = (focal * yp) + py;

        // The error is the squared euclidian distance between the predicted and observed position.
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);

        return true;

    }

    double observed_x, observed_y, cx, cy, t_, n1_, n2_, n3_, z0_;
    int num_cams;

};

// Refractive Reprojection Error function for planes
class refractiveReprojError {

 public:

 refractiveReprojError(double observed_x, double observed_y, double cx, double cy, int num_cams, double t, double n1, double n2, double n3, double z0, int gridx, int gridy, double grid_phys, int index, int plane_id)
     : observed_x(observed_x), observed_y(observed_y), cx(cx), cy(cy), num_cams(num_cams), t_(t), n1_(n1), n2_(n2), n3_(n3), z0_(z0), gridx_(gridx), gridy_(gridy), grid_phys_(grid_phys), index_(index), plane_id_(plane_id) { }

    template <typename T>
        bool operator()(const T* const camera,
                        const T* const plane,
                        T* residuals) const {

        // Inital guess for points on glass
        T* R = new T[9];
        ceres::AngleAxisToRotationMatrix(camera, R);

        T c[3];
        for (int i=0; i<3; i++) {
            c[i] = T(0);
            for (int j=0; j<3; j++) {
                c[i] += -R[i*1 + j*3]*camera[j+3];
            }
        }

        // Generate point on grid
        int number = index_ - plane_id_*(gridx_*gridy_);
        int j = floor(number/gridx_);
        int i = number - j*gridx_;



        T* grid_point = new T[3];
        grid_point[0] = T(i*grid_phys_); grid_point[1] = T(j*grid_phys_); grid_point[2] = T(0);

        // Move point to be on given grid plane
        T point[3];
        ceres::AngleAxisRotatePoint(plane, grid_point, point);
        point[0] += plane[3]; point[1] += plane[4]; point[2] += plane[5];

        // Solve for refraction to reproject point into camera
        T* a = new T[3];
        T* b = new T[3];
        a[0] = c[0] + (point[0]-c[0])*(T(-t_)+T(z0_)-c[2])/(point[2]-c[2]);
        a[1] = c[1] + (point[1]-c[1])*(T(-t_)+T(z0_)-c[2])/(point[2]-c[2]);
        a[2] = T(-t_)+T(z0_);
        b[0] = c[0] + (point[0]-c[0])*(T(z0_)-c[2])/(point[2]-c[2]);
        b[1] = c[1] + (point[1]-c[1])*(T(z0_)-c[2])/(point[2]-c[2]);
        b[2] = T(z0_);

        T rp = sqrt( pow(point[0]-c[0],2) + pow(point[1]-c[1],2) );
        T dp = point[2]-b[2];
        T phi = atan2(point[1]-c[1],point[0]-c[0]);

        T ra = sqrt( pow(a[0]-c[0],2) + pow(a[1]-c[1],2) );
        T rb = sqrt( pow(b[0]-c[0],2) + pow(b[1]-c[1],2) );
        T da = a[2]-c[2];
        T db = b[2]-a[2];

        T f, g, dfdra, dfdrb, dgdra, dgdrb;

        // Newton Raphson loop to solve for Snell's law
        for (int iter=0; iter<20; iter++) {

            f = ( ra/sqrt(pow(ra,2)+pow(da,2)) ) - ( T(n2_/n1_)*(rb-ra)/sqrt(pow(rb-ra,2)+pow(db,2)) );
            g = ( (rb-ra)/sqrt(pow(rb-ra,2)+pow(db,2)) ) - ( T(n3_/n2_)*(rp-rb)/sqrt(pow(rp-rb,2)+pow(dp,2)) );

            dfdra = ( T(1.0)/sqrt(pow(ra,2)+pow(da,2)) )
                - ( pow(ra,2)/pow(pow(ra,2)+pow(da,2),1.5) )
                + ( T(n2_/n1_)/sqrt(pow(ra-rb,2)+pow(db,2)) )
                - ( T(n2_/n1_)*(ra-rb)*(T(2)*ra-T(2)*rb)/(T(2)*pow(pow(ra-rb,2)+pow(db,2),1.5)) );

            dfdrb = ( T(n2_/n1_)*(ra-rb)*(T(2)*ra-T(2)*rb)/(T(2)*pow(pow(ra-rb,2)+pow(db,2),1.5)) )
                - ( T(n2_/n1_)/sqrt(pow(ra-rb,2)+pow(db,2)) );

            dgdra = ( (ra-rb)*(T(2)*ra-T(2)*rb)/(T(2)*pow(pow(ra-rb,2)+pow(db,2),1.5)) )
                - ( T(1.0)/sqrt(pow(ra-rb,2)+pow(db,2)) );

            dgdrb = ( T(1.0)/sqrt(pow(ra-rb,2)+pow(db,2)) )
                + ( T(n3_/n2_)/sqrt(pow(rb-rp,2)+pow(dp,2)) )
                - ( (ra-rb)*(T(2)*ra-T(2)*rb)/(T(2)*pow(pow(ra-rb,2)+pow(db,2),1.5)) )
                - ( T(n3_/n2_)*(rb-rp)*(T(2)*rb-T(2)*rp)/(T(2)*pow(pow(rb-rp,2)+pow(dp,2),1.5)) );

            ra = ra - ( (f*dgdrb - g*dfdrb)/(dfdra*dgdrb - dfdrb*dgdra) );
            rb = rb - ( (g*dfdra - f*dgdra)/(dfdra*dgdrb - dfdrb*dgdra) );

                }

        a[0] = ra*cos(phi) + c[0];
        a[1] = ra*sin(phi) + c[1];

        // Continuing projecting point a to camera
        T p[3];
        ceres::AngleAxisRotatePoint(camera, a, p);

        // camera[3,4,5] are the translation.
        p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];

        // Compute the center of distortion.
        T xp = p[0] / p[2]; T yp = p[1] / p[2];

        // Image principal points
        T px = T(cx); T py = T(cy);

        // Apply second and fourth order radial distortion.
        const T& l1 = camera[7];
        const T& l2 = camera[8];
        T r2 = xp*xp + yp*yp;
        //T distortion = T(1.0) + r2  * (l1 + l2  * r2);

        // Compute final projected point position.
        const T& focal = camera[6];
        T predicted_x = (focal * xp) + px;
        T predicted_y = (focal * yp) + py;

        // The error is the squared euclidian distance between the predicted and observed position.
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);

        return true;

    }

    double observed_x, observed_y, cx, cy, t_, n1_, n2_, n3_, z0_, grid_phys_;
    int num_cams, gridx_, gridy_, index_, plane_id_;

};

// Relaxation Tracking Error function
// class rlxTrackingError {

//  public:

//  rlxTrackingError(string path):
//     path_(path) {}

//     template <typename T>
//     bool operator()(const T* const params,
//                     T* residuals) const {

//         pTracking track(path_, params[0], params[1]);

//         track.set_vars(double(params[0]),
//                         double(params[1]),
//                         double(params[2]),
//                         double(params[3]));

//         track.track_frames(15, 16);
//         residuals[0] = T(1.0) - T(track.sim_performance());

//         return true;

//     }

//     string path_;

// };

// FUNCTION DEFINITIONS

//double BA_pinhole(baProblem &ba_problem, string ba_file, Size img_size, vector<int> const_points);

//double BA_refractive(baProblem_ref &ba_problem, string ba_file, Size img_size, vector<int> const_points);

#endif
