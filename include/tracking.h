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

#include "std_include.h"
#include "refocusing.h"
#include "typedefs.h"

using namespace std;
using namespace cv;

class pTracking {

 public:
    ~pTracking() {

    }

 pTracking(string particle_file, double R_n, double R_s): path_(particle_file), R_s(R_s), R_n(R_n) {
        initialize();
    }
    
    void initialize();
    void read_points();

    void track_all();
    vector<Point2i> track_frame(int f1, int f2);

    void plot_complete_paths();

    void write_quiver_data(int frame, string path);

 private:

    // Functions
    int find_matches(vector<Mat> Pij, vector<Mat> Pi, vector< vector<int> > S_r, vector< vector<int> > S_c, vector<Point2i> &matches);
    double update_probabilities(vector<Mat> &Pij, vector<Mat> &Pi, vector<Mat> &Pij2, vector<Mat> &Pi2);
    void normalize_probabilites(vector<Mat> &Pij, vector<Mat> &Pi);
    void build_probability_sets(vector< vector<int> > S_r, vector< vector<int> > S_c, vector<Mat> &Pij, vector<Mat> &Pi, vector<Mat> &Pij2, vector<Mat> &Pi2);
    void build_relaxation_sets(int frame1, int frame2, vector< vector<int> > S_r, vector< vector<int> > S_c, double C, double D, double E, double F, vector< vector< vector< vector<Point2i> > > > &theta);
    vector< vector<int> > neighbor_set(int frame1, int frame2, double r);
    vector<int> points_in_region(int frame, Point3f center, double r);

    // Variables
    string path_;

    vector<Point3f> points_;
    vector< vector<Point3f> > all_points_;
    vector<volume> vols_;
    
    vector< vector<Point2i> > all_matches;

    double R_n, R_s, V_n, V_s;
    int N;
    double tol;

};

#endif
