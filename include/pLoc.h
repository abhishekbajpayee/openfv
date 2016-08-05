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

// -------------------------------------------------------
// -------------------------------------------------------
// Synthetic Aperture - Particle Tracking Velocimetry Code
// --- Particle Localization Header ---
// -------------------------------------------------------
// Author: Abhishek Bajpayee
//         Dept. of Mechanical Engineering
//         Massachusetts Institute of Technology
// -------------------------------------------------------
// -------------------------------------------------------

#ifndef LOCALIZATION_LIBRARY
#define LOCALIZATION_LIBRARY

#include "std_include.h"
#include "refocusing.h"
#include "typedefs.h"

using namespace std;
using namespace cv;

/*!
    Class with functions that allow a user to find particles in a refocused volume.
*/
class pLocalize {

 public:

    ~pLocalize() {

    }

    /*! pLocalize constructor
        \param s A localizer_settings struct variable containing all relevant settings for the localizer
        \param refocus An saRefocus object that has been initialized with data etc.
        \param s2 A refocus_settings struct variable. This is the same variable that is passed to the saRefocus class constructor.
    */
    pLocalize(localizer_settings s, saRefocus refocus, refocus_settings s2);

    /*! Return detected particles */
    vector<Point3f> detected_particles() { return particles_; }

    void run();

    /*! Find particles in all in all frames in the sequence. This is like calling the find_particles_3d() function for each frame.
    */
    void find_particles_all_frames();
    /*! Find particles in the volume. The depth extents between which particles are found are specified in the localizer settings.
        \param frame The frame number in which to find the particles.
    */
    void find_particles_3d(int frame);
    
    void save_refocus(int frame);
    void z_resolution();
    void crop_focus();

    void find_clusters();
    void clean_clusters();
    void collapse_clusters();

    void find_particles(Mat image, vector<Point2f> &points_out);
    void refine_subpixel(Mat image, vector<Point2f> points_in, vector<particle2d> &points_out);

    /*! Write particles to file
        \param path Path of file to write particles to
    */
    void write_all_particles_to_file(string path);
    /*! Write particles to a folder. This is different from the write_all_particles_to_file() function in that this accepts a path to a directory and write particles in a file inside a folder called particles in the path. Note that the path and a folder called particles must exist. The filename is determined automatically based on the relevant settings being used.
        \param path Path of folder to write file in. This path must contain another folder called particles in it.
    */
    void write_all_particles(string path);
    void write_particles_to_file(string path);
    void write_clusters(vector<particle2d> &particles3D_, string path);
    void draw_points(Mat image, Mat &drawn, vector<Point2f> points);
    void draw_points(Mat image, Mat &drawn, vector<particle2d> points);
    void draw_point(Mat image, Mat &drawn, Point2f point);

 private:

    int point_in_list(Point2f point, vector<Point2f> points);
    double min_dist(Point2f point, vector<Point2f> points);
    double get_zloc(vector<particle2d> cluster);

    int window_;
    int cluster_size_;
    double zmin_;
    double zmax_;
    double dz_;
    double thresh_;
    double zext_;

    int zmethod_;

    vector<particle2d> particles3D_;
    vector< vector<particle2d> > clusters_;
    vector<Point3f> particles_;
    vector< vector<Point3f> > particles_all_;

    saRefocus refocus_;
    refocus_settings s2_;

    int show_particles_;

};

#endif
