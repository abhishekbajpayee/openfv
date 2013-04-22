// -------------------------------------------------------
// -------------------------------------------------------
// Synthetic Aperture - Particle Tracking Velocimetry Code
// --- Calibration IO Library Header ---
// -------------------------------------------------------
// Author: Abhishek Bajpayee
//         Dept. of Mechanical Engineering
//         Massachusetts Institute of Technology
// -------------------------------------------------------
// -------------------------------------------------------

#ifndef CALIBRATION_IO_LIB
#define CALIBRATION_IO_LIB

#include "std_include.h"
#include "optimization.h"

using namespace cv;
using namespace std;

// FUNCTIONS

<<<<<<< HEAD
void disp_M(Mat M);

=======
vector<string> read_cam_names(string path);

vector< vector<Mat> > read_calib_imgs(string path, vector<string> cam_names);

void disp_M(Mat M);

void write_point_data(vector< vector< vector<Point2f> > > all_corner_points, vector<Mat> cameraMats, vector< vector<Mat> > rvecs, vector< vector<Mat> > tvecs, vector<Mat> dist_coeffs, string filename);

>>>>>>> origin/master
void write_ba_result(baProblem& ba_problem, string filename);

void write_ba_matlab(baProblem& ba_problem, vector<Mat> translations, vector<Mat> P_mats);

<<<<<<< HEAD
void write_align_data(vector<Mat> rvecs, vector<Mat> tvecs, string align_file);

void write_aligned(alignProblem align_problem, string aligned_file);

void write_calib_result(baProblem ba_problem, vector<string> cam_names, Size img_size, double pix_per_phys, string path);
=======
void write_align_data(baProblem &ba_problem, string filename, vector< vector<Mat> > translations_new, char* argv);
>>>>>>> origin/master

#endif
