// -------------------------------------------------------
// -------------------------------------------------------
// Synthetic Aperture - Particle Tracking Velocimetry Code
// --- Refocusing IO Library Header ---
// -------------------------------------------------------
// Author: Abhishek Bajpayee
//         Dept. of Mechanical Engineering
//         Massachusetts Institute of Technology
// -------------------------------------------------------
// -------------------------------------------------------

#ifndef REFOCUSING_IO_LIB
#define REFOCUSING_IO_LIB

#include "std_include.h"

using namespace cv;
using namespace std;

<<<<<<< HEAD
void read_calib_data(string filename, vector<Mat> &P_mats, vector<Mat> &rvecs, vector<Mat> &tvecs, vector<Mat> &translations, double &scale);
=======
void read_calib_data(string filename, vector<Mat> &P_mats, vector<Mat> &translations, double &scale);
>>>>>>> origin/master

void read_refocusing_imgs(string path, vector<string> cam_names, vector< vector<Mat> > &refocusing_imgs);

#endif
