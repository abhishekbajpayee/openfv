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
// --- Tracking Library Header ---
// -------------------------------------------------------
// Author: Abhishek Bajpayee
//         Dept. of Mechanical Engineering
//         Massachusetts Institute of Technology
// -------------------------------------------------------
// -------------------------------------------------------

#ifndef TRACKING_LIBRARY
#define TRACKING_LIBRARY

// Ceres Solver headers
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "std_include.h"
#include "typedefs.h"
#include "tools.h"

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
    int show_refocused_;

};

/*!
    Class with functions to track particles using relaxation based method.
*/
class pTracking {

 public:
    ~pTracking() {

    }

    /*! pTracking constructor
        \param particle_file Path of file containing list of particles
        \param Rn Neighborhood threshold
        \param Rs Search threshold
    */
    pTracking(string particle_file, double Rn, double Rs);

    /*! Function to reset relevant variables
        \param method
        \param rn Neighborhood threshold
        \param rs Search threshold
        \param e Relaxation set parameter E from [1]
        \param f Relaxation set parameter F from [1]
    */
    void set_vars(int method, double rn, double rs, double e, double f);

    void initialize();
    void read_points();

    /*! Track particles between given frames
        \param start Starting frame
        \param end Ending frame
    */
    void track_frames(int start, int end);
    /*! Track particles over all the frames in the input file */
    void track_all();
    vector<Point2i> track_frame(int f1, int f2, int &count);
    void track_frame_n(int f1, int f2);

    /*! Find paths longer than a certain length l in time frames */
    void find_long_paths(int l);
    /*! Find paths exactly of length l in time frames */
    void find_sized_paths(int l);
    // void plot_long_paths();
    // void plot_sized_paths();
    // void plot_all_paths();

    void write_quiver_data();

    /*! Write tracking results to file. The output filename and location is automatically
      generated based on the path and name of the input particle file passed to the
      pTracking constructor. For example, if the particle_file is
      ``/home/user/project/particles.txt`` and the prefix string passed is ``prefix`` then
      then tracking results are saved in ``/home/user/project/particles_prefix_result.txt``.
      The format of this results file is:
      \verbatim embed:rst
      .. code ::

          <number of time frames>
          <number of matches in frame 1>
          <index of particle in frame 1>TAB<index of match particle in frame 2>
          <index of particle in frame 1>TAB<index of match particle in frame 2>
          ...
          <number of matches in frame 2>
          <index of particle in frame 2>TAB<index of match particle in frame 3>
          <index of particle in frame 2>TAB<index of match particle in frame 3>
          ...
          ...
      \endverbatim
      where the "\t" between particle indices is a TAB character and the indices are of
      particles in the particles file used.
      \param prefix String of text to add to the output filename
    */
    void write_tracking_result(string prefix);
    void write_all_paths(string path);
    void write_long_quiver(string path, int l);

    double sim_performance();

    vector<int> get_match_counts();
    Mat getP();

 private:

    // Functions
    int find_matches(vector<Mat> Pij, vector<Mat> Pi, vector< vector<int> > S_r, vector< vector<int> > S_c, vector<Point2i> &matches);

    double update_probabilities(vector<Mat> &Pij, vector<Mat> &Pi, vector<Mat> &Pij2, vector<Mat> &Pi2);
    void normalize_probabilites(vector<Mat> &Pij, vector<Mat> &Pi);
    void build_probability_sets(vector< vector<int> > S_r, vector< vector<int> > S_c, vector<Mat> &Pij, vector<Mat> &Pi, vector<Mat> &Pij2, vector<Mat> &Pi2);

    // New functions
    int find_matches_n(Mat Pij, Mat Pi, vector<Point2i> &matches);
    double update_probabilities_n(Mat &Pij, Mat &Pi, Mat &Pij2, Mat &Pi2);
    void normalize_probabilites_n(Mat &Pij, Mat &Pi);
    void build_probability_sets_n(vector< vector<int> > S_r, vector< vector<int> > S_c, Mat &Pij, Mat &Pi, Mat &Pij2, Mat &Pi2);
    void build_relaxation_sets_n(int frame1, int frame2, vector< vector<int> > S_r, vector< vector<int> > S_c, double C, double D, double E, double F, vector< vector< vector<Point2i> > > &theta);

    void build_relaxation_sets(int frame1, int frame2, vector< vector<int> > S_r, vector< vector<int> > S_c, double C, double D, double E, double F, vector< vector< vector< vector<Point2i> > > > &theta);
    vector< vector<int> > neighbor_set(int frame, double r);
    vector< vector<int> > candidate_set(int frame1, int frame2, double r);
    vector<int> points_in_region(int frame, Point3f center, double r);

    bool is_used(vector< vector<int> > used, int k, int i);

    // Variables
    string path_;

    vector<Point3f> points_;
    vector< vector<Point3f> > all_points_;
    vector<volume> vols_;

    vector< vector<Point2i> > all_matches;
    vector<int> match_counts;

    vector< vector<int> > long_paths_;
    vector<particle_path> sized_paths_;

    double R_n, R_s, V_n, V_s;
    double A, B, C, D, E, F;
    int N;
    double tol;
    int offset;

    Mat P_;

    int method_;
    int reject_singles_;

};

// Poly2 Fit Error function
class poly2FitError {

 public:

 poly2FitError(double xin, double yin):
    x(xin), y(yin) {}

    template <typename T>
    bool operator()(const T* const params,
                    T* residuals) const {

        residuals[0] = y - T(params[0]*x*x + params[1]*x + params[2]);
        return true;

    }

    double x;
    double y;

};

// Gaussian Fit Error function
class gaussFitError {

 public:

 gaussFitError(double xin, double yin):
    x(xin), y(yin) {}

    template <typename T>
    bool operator()(const T* const params,
                    T* residuals) const {

        residuals[0] = y - T(params[0]*exp(-pow(((x-params[1])/params[2]),2)));
        return true;

    }

    double x;
    double y;

};


#endif
