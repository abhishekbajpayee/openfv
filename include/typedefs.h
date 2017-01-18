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

/*! Settings container passed to saRefocus constructor. */
struct refocus_settings {
    
    //! Flag indicating if data is in mtiff format files
    int mtiff; // 1 for using multipage tiffs

    //! Flag to use multiplicative refocusing
    int mult; // 1 for Multiplicative
    //! Multiplicative exponent
    double mult_exp;

    //! Flag to use a GPU or not
    int use_gpu; // 1 for GPU
    //! Use Homography Fit (HF) method or not
    int hf_method; // 1 to use corner method

    //! Path to calibration data file
    string calib_file_path;
    //! Path to directory where images to be refocused are
    string images_path;
  
    // string frames;
    //! Flag indicating if all frames in the input data should be processed
    int all_frames;
    //! Start frame number (used if all_frames is set to 0)
    int start_frame;
    //! End frame number (used if all_frames is set to 0)
    int end_frame;
    //! Successive frames to skip (used if all_frames is set to 0)
    int skip;
    
};

struct calibration_settings {

    //! Path to directory where input images / videos lie
    string images_path;
    //! Number of corners in grid (horizontal x vertical)
    Size grid_size;
    //! Physical size of grid in [mm]
    double grid_size_phys;

    //! Flag indicating if calibration refractive or not
    int refractive;
    //! Flag indicating if calibration data is in mtiff files or not
    int mtiff;
    //! Flag indicating if calibration data is in mp4 files or not
    int mp4;

    //! Frames to skip between successive reads
    int skip;
    //! Number of the frame to start reading at
    int start_frame;
    //! Number of the frame to end reading at
    int end_frame;

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

/*! Settings container passed to pLocalize constructor. */
struct localizer_settings {
    
    /*! Window size to use around a given pixel
      to search for particle peak */
    int window;
    //! Starting z depth of particle search region
    double zmin;
    //! Ending z depth of particle search region
    double zmax;
    //! dz between each successive refocused depth
    double dz;
    //! Amount by which to threshold the refocused images
    double thresh;
    //! Method used to calculate the z coordinate of a particle cluster. 1 is for mean z location, 2 is for a quadratic fit (poly2) and 3 is for a gaussian fit.
    int zmethod;
    int show_particles;
    int show_refocused;
    //! Critical cluster size above which to consider a cluster a particle. As of now, regardless of cluster size, a maximum physical z extent of 2.5 mm is used to determine is a detected 2D particle is part of a 3D particle cluster or not.
    int cluster_size;
    
};

// TODO: consider making size a fractional where max intensity
//       defines how much of a pixel to count
/*! Data type for a particle in 2D. Carries 3D location, 
    average intensity and size in pixels. Used during
    particle localization. */
struct particle2d {

    //! x coordinate of particle
    double x;
    //! y coordinate of particle
    double y;
    //! z coordinate of particle
    double z;
    //! Average intensity of particle
    double I;
    //! Size of particle in pixels
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

struct particle_path {
    
    int start_frame;
    vector<int> path;
    
};

/*
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

struct voxel {
    int x;
    int y;
    int z;
    double I;
};
*/

#endif
