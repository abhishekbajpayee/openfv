#ifndef DATA_TYPES
#define DATA_TYPES

#include "std_include.h"

using namespace std;
using namespace cv;

const double pi = 3.14159;

struct refocus_settings {

    int mult; // 1 for Multiplicative
    double mult_exp;
    int gpu; // 1 for GPU
    int ref; // 1 for refractive
    int corner_method; // 1 to use corner method

    string calib_file_path;

    string images_path;
    int mtiff; // 1 for using multipage tiffs
    int start_frame;
    int end_frame;
    int upload_frame;
    int all_frames;
    
    int preprocess;
    string preprocess_file;

    double zmin, zmax, dz, thresh;
    string save_path;

};

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

struct localizer_settings {
    
    int window;
    double zmin, zmax, dz;
    double thresh;
    int zmethod;
    int show_particles;
    
};

struct particle_path {
    
    int start_frame;
    vector<int> path;
    
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
    double v;
};

struct voxel {
    int x;
    int y;
    int z;
    double I;
};

#endif
