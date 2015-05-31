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

#ifndef DATA_TYPES
#define DATA_TYPES

#include "std_include.h"

using namespace std;
using namespace cv;

const double pi = 3.14159;

struct refocus_settings {

    int mult; // 1 for Multiplicative
    double mult_exp;
    int gpu; // 1 for GPU
    int ref; // 1 for refractive
    int corner_method; // 1 to use corner method

    string calib_file_path;

    string images_path;
    int mtiff; // 1 for using multipage tiffs
    int start_frame;
    int end_frame;
    int upload_frame;
    int all_frames;
    
    int preprocess;
    string preprocess_file;

    double zmin, zmax, dz, thresh;
    string save_path;

};

struct safe_refocus_settings {

  double dp;
  double minDist; 
  double param1;
  double param2;
  int minRadius;
  int maxRadius;
  int gKerWid;
  int gKerHt;
  int gKerSigX;
  int gKerSigY;
  int circle_rim_thickness;
  int debug;

};

struct refocusing_data {
    vector<Mat> P_mats_u; // Unaligned P matrices
    vector<Mat> P_mats;
    vector<string> cam_names;
    double scale;
    Size img_size;
    int num_cams;
    double warp_factor;
};

struct refocusing_data_ref {
    vector<Mat> P_mats_u; // Unaligned P matrices
    vector<Mat> P_mats;
    vector<string> cam_names;
    double scale;
    Size img_size;
    int num_cams;
    int n1;
    int n2;
    int n3;
    double zW;
};

struct localizer_settings {
    
    int window;
    double zmin, zmax, dz;
    double thresh;
    int zmethod;
    int show_particles;
    int cluster_size;
    
};

struct particle_path {
    
    int start_frame;
    vector<int> path;
    
};

// Data type for a particle in 2D:
// carries 3D location, average intensity and size in pixels
// TODO: consider making size a fractional where max intensity
//       defines how much of a pixel to count
struct particle2d {
    double x;
    double y;
    double z;
    double I;
    int size;
};

// Data type to store the bounds of a volume
// in which particles in a given frame are
// contained
struct volume {
    double x1;
    double x2;
    double y1;
    double y2;
    double z1;
    double z2;
    double v;
};

struct voxel {
    int x;
    int y;
    int z;
    double I;
};

#endif
