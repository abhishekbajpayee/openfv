#ifndef DATA_TYPES
#define DATA_TYPES

#include "std_include.h"

struct refocusing_data {
    vector<Mat> P_mats_u; // Unaligned P matrices
    vector<Mat> P_mats;
    vector<string> cam_names;
    double scale;
    Size img_size;
    int num_cams;
};

#endif
