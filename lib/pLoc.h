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
#include "typedefs.h"

using namespace std;
using namespace cv;

class pLocalize {

 public:

    ~pLocalize() {

    }

 pLocalize(int window):
    window_(window) {}

    void find_particles_3d();

    void find_particles(Mat image, vector<Point2f> &points_out);
    void refine_subpixel(Mat image, vector<Point2f> points_in, vector<particle2d> &points_out);

    void find_clusters();
    void clean_clusters();

    void draw_points(Mat image, Mat &drawn, vector<Point2f> points);
    void draw_point(Mat image, Mat &drawn, Point2f point);

 private:

    int point_in_list(Point2f point, vector<Point2f> points);
    double min_dist(Point2f point, vector<Point2f> points);

    int window_;

};

#endif
