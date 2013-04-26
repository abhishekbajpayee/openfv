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

class pLocalize {

 public:

    ~pLocalize() {

    }

 pLocalize(int window, double zmin, double zmax, double dz, double thresh, int cluster_size, saRefocus refocus):
    window_(window), zmin_(zmin), zmax_(zmax), dz_(dz), thresh_(thresh), cluster_size_(cluster_size), refocus_(refocus) {}

    vector<Point3f> detected_particles() { return particles_; }

    void run();

    void find_particles_3d();
    void find_clusters();
    void clean_clusters();
    void collapse_clusters();

    void find_particles(Mat image, vector<Point2f> &points_out);
    void refine_subpixel(Mat image, vector<Point2f> points_in, vector<particle2d> &points_out);

    void draw_points(Mat image, Mat &drawn, vector<Point2f> points);
    void draw_point(Mat image, Mat &drawn, Point2f point);

 private:

    int point_in_list(Point2f point, vector<Point2f> points);
    double min_dist(Point2f point, vector<Point2f> points);

    int window_;
    int cluster_size_;
    double zmin_;
    double zmax_;
    double dz_;
    double thresh_;

    vector<particle2d> particles3D_;
    vector< vector<particle2d> > clusters_;
    vector<Point3f> particles_;

    saRefocus refocus_;

};

#endif
