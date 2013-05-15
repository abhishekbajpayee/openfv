// -------------------------------------------------------
// -------------------------------------------------------
// Synthetic Aperture - Particle Tracking Velocimetry Code
// --- Refocusing Library Header ---
// -------------------------------------------------------
// Author: Abhishek Bajpayee
//         Dept. of Mechanical Engineering
//         Massachusetts Institute of Technology
// -------------------------------------------------------
// -------------------------------------------------------

#ifndef REFOCUSING_LIBRARY
#define REFOCUSING_LIBRARY

#include "std_include.h"
#include "calibration.h"
#include "typedefs.h"

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace std;
using namespace cv;

class saRefocus {

 public:
    ~saRefocus() {

    }

 saRefocus(refocusing_data refocusing_params, int frame):
    P_mats_(refocusing_params.P_mats), P_mats_u_(refocusing_params.P_mats_u), cam_names_(refocusing_params.cam_names), img_size_(refocusing_params.img_size), scale_(refocusing_params.scale), num_cams_(refocusing_params.num_cams), z(0), thresh(0), frame_(frame) {}

    int num_cams() { return num_cams_; }
    double scale() { return scale_; }
    Size img_size() { return img_size_; }
    int num_frames() { return imgs[0].size(); }

    void read_imgs(string path);

    void GPUliveView();
    void initializeGPU();
    void uploadToGPU();
    void GPUrefocus(double z, double thresh, int live, int frame);

    Mat result;

 private:

    // data types and private functions
    vector<Mat> P_mats_;
    vector<Mat> P_mats_u_;
    vector<string> cam_names_;
    Size img_size_;
    double scale_;
    int num_cams_;

    double z;
    double thresh;
    int frame_;

    Mat refocused_host_;

    // Vector of vectors that stores images from all cameras and for all time steps
    vector< vector<Mat> > imgs;

    vector<gpu::GpuMat> array;
    vector< vector<gpu::GpuMat> > array_all;
    gpu::GpuMat temp;
    gpu::GpuMat temp2;
    gpu::GpuMat refocused;
    
    int frame_to_upload_;

};

#endif
