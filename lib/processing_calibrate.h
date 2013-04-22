#ifndef CALIBRATION_PROC_LIB
#define CALIBRATION_PROC_LIB

#include "std_include.h"
<<<<<<< HEAD
#include "optimization.h"
=======
>>>>>>> origin/master

using namespace cv;
using namespace std;

// FUNCTIONS

<<<<<<< HEAD
=======
double reprojection_errors(const vector<vector<Point3f> > &objectPoints,
                           const vector<vector<Point2f> > &imagePoints,
                           const vector<Mat> &rvecs, const vector<Mat> &tvecs,
                           const Mat &cameraMatrix , const Mat &distCoeffs,
                           vector<float> &perViewErrors);

void all_reprojection_errors(vector<double> &reproj_errors, vector< vector<float> > &per_view_errors,
                             vector< vector<Point3f> > all_pattern_points, vector< vector< vector<Point2f> > > all_corner_points,
                             int num_cams, int num_imgs, vector<Mat> cameraMats, vector<Mat> dist_coeffs, 
                             vector< vector<Mat> > rvecs, vector< vector<Mat> > tvecs);

vector< vector< vector<Point2f> > > find_corners(vector< vector<Mat> > calib_imgs, Size grid_size, int draw);

void calibrate_cameras(vector< vector<Point3f> > &all_pattern_points, vector< vector< vector<Point2f> > > corner_points, 
                       int num_cams, int num_imgs, Size grid_size, Size img_size, vector<Mat> &cameraMats, 
                       vector<Mat> &dist_coeffs, vector< vector<Mat> > &rvecs, vector< vector<Mat> > &tvecs);

>>>>>>> origin/master
void get_translations_all(vector< vector<Mat> > rvecs, vector< vector<Mat> > tvecs, 
                     vector< vector<Mat> > &translations);

void get_translations(vector<Mat> rvecs, vector<Mat> tvecs, vector<Mat> &translations);

<<<<<<< HEAD
Mat P_from_KRT(Mat K, Mat rvec, Mat tvec, Mat rmean);

=======
>>>>>>> origin/master
#endif
