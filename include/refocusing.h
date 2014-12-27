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
#include "cuda_lib.h"

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace std;
using namespace cv;

class saRefocus {

 public:
    ~saRefocus() {

    }

    saRefocus();

 saRefocus(refocusing_data refocusing_params, int frame, int mult, double mult_exp):
    P_mats_(refocusing_params.P_mats), P_mats_u_(refocusing_params.P_mats_u), cam_names_(refocusing_params.cam_names), img_size_(refocusing_params.img_size), scale_(refocusing_params.scale), num_cams_(refocusing_params.num_cams), warp_factor_(refocusing_params.warp_factor), z_(0), thresh(0), frame_(frame), mult_(mult), mult_exp_(mult_exp) { }

    saRefocus(refocus_settings settings);

    saRefocus(int num_cams, double f);

    int num_cams() { return num_cams_; }
    double scale() { return scale_; }
    Size img_size() { return img_size_; }
    int num_frames() { return imgs[0].size(); }

    void read_calib_data(string path);
    void read_calib_data_pin(string path);

    void read_imgs(string path);
    void read_imgs_mtiff(string path);

    void CPUliveView();
    void GPUliveView();
    void initializeGPU();

    Mat refocus(double z, double rx, double ry, double rz, double thresh, int frame);

    void GPUrefocus(double thresh, int live, int frame);
    void GPUrefocus_ref(double thresh, int live, int frame);
    void GPUrefocus_ref_corner(double thresh, int live, int frame);

    void CPUrefocus(double z, double thresh, int live, int frame);
    void CPUrefocus_ref(double z, double thresh, int live, int frame);
    void CPUrefocus_ref_corner(double z, double thresh, int live, int frame);

    void dump_stack(string path, double zmin, double zmax, double dz, double thresh, string type);
    void calculateQ(double zmin, double zmax, double dz, double thresh, int frame, string refPath);
    void return_stack(double zmin, double zmax, double dz, double thresh, int frame, vector<Mat> &stack);
    double getQ(vector<Mat> &stack, vector<Mat> &refStack);

    // Expert mode functions
    void setArrayData(vector<Mat> imgs, vector<Mat> Pmats, vector<Mat> cam_locations);
    void addView(Mat img, Mat P, Mat location);
    void setF(double f);
    void setMult(int flag, double exp);
    string showSettings();

 private:

    void uploadToGPU();
    void uploadToGPU_ref();

    void preprocess(Mat in, Mat &out);
    void parse_preprocess_settings(string file);

    void calc_ref_refocus_map(Mat_<double> Xcam, double z, Mat_<double> &x, Mat_<double> &y, int cam);
    void calc_refocus_map(Mat_<double> &x, Mat_<double> &y, int cam);
    void calc_ref_refocus_H(Mat_<double> Xcam, double z, int cam, Mat &H);
    void calc_refocus_H(int cam, Mat &H);
    void img_refrac(Mat_<double> Xcam, Mat_<double> X, Mat_<double> &X_out);

    void adaptiveNorm(Mat in, Mat &out, int xf, int yf);
    void slidingMinToZero(Mat in, Mat &out, int xf, int yf);

    // Refocusing result
    Mat result_;

    // data types and private functions
    vector<Mat> P_mats_;
    vector<Mat> P_mats_u_;
    vector<string> cam_names_;
    vector<Mat> cam_locations_;
    Size img_size_;
    double scale_;
    int num_cams_;
    
    // Scene geometry params
    float geom[5];

    double z_, xs_, ys_, zs_, rx_, ry_, rz_, cxs_, cys_, czs_, crx_, cry_, crz_;
    double thresh;
    int frame_;
    vector<int> frames_;
    double mult_exp_;
    double warp_factor_;
    int active_frame_;

    Mat refocused_host_;

    // Vector of vectors that stores images from all cameras and for all time steps
    vector< vector<Mat> > imgs;
    Mat cputemp; Mat cputemp2; Mat cpurefocused;

    vector<gpu::GpuMat> array;
    vector<gpu::GpuMat> P_mats_gpu;
    vector<gpu::GpuMat> cam_locations_gpu;
    vector<gpu::GpuMat> xmaps, ymaps;
    vector< vector<gpu::GpuMat> > array_all;
    gpu::GpuMat temp, temp2, refocused, xmap, ymap;
    
    int frame_to_upload_;
    int mult_;

    int GPU_FLAG;
    int REF_FLAG;
    int CORNER_FLAG; // Flag to use corner based homography fit method
    int MTIFF_FLAG;
    int ALL_FRAME_FLAG;
    int INVERT_Y_FLAG;
    int EXPERT_FLAG;
    
    int preprocess_;
    vector<int> pp_ops;
    vector<int> thresh_vals;
    vector<int> gbkernel;
    vector<float> gbsigma;
    vector<int> anwx;
    vector<int> anwy;
    vector<int> mfkernel;
    vector<int> sMeankernel;
    vector<int> smtzwx;
    vector<int> smtzwy;

};

#endif
