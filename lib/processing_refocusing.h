#ifndef REFOCUSING_PROC_LIB
#define REFOCUSING_PROC_LIB

#include "std_include.h"

using namespace cv;
using namespace std;

<<<<<<< HEAD
void T_from_P(Mat P_mats, Mat &H, double z, double scale, Size img_size);

void refocus_img(vector< vector<Mat> > refocusing_imgs, vector<Mat> P_mats, double pix_per_phys, double z, int method);
=======
double get_grid_size_pix(vector< vector< vector<Point2f> > > all_corner_points, Size grid_size, int center_cam_id);

void T_from_P(Mat P_mats, Mat &H, double z, double scale, Size img_size);

void refocus_img(vector< vector<Mat> > refocusing_imgs, vector<Mat> P_mats, double pix_per_phys, double z);
>>>>>>> origin/master

void get_P_mats(vector<Mat> cameraMats, vector< vector<Mat> > rvecs, vector< vector<Mat> > tvecs, vector<Mat> &P_mats);

void get_H_mats(vector< vector<Mat> > rvecs, vector< vector<Mat> > tvecs, vector< vector<Mat> > translations, double scale, vector<Mat> &H_mats);

#endif
