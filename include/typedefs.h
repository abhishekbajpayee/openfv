#ifndef DATA_TYPES
#define DATA_TYPES

#include "std_include.h"

using namespace std;
using namespace cv;

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

// Data type for a particle in 2D:
// carries 3D location, average intensity and size in pixels
// TODO: consider making size a fractional where max intensity
//       defines how much of a pixel to count
struct particle2d {
    double x;
    double y;
    double z;
    int I;
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
};

#endif
