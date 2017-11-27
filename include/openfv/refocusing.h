//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//                           License Agreement
//                For Open Source Flow Visualization Library
//
// Copyright 2013-2017 Abhishek Bajpayee
//
// This file is part of OpenFV.
//
// OpenFV is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License version 2 as published by the Free Software Foundation.
//
// OpenFV is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License version 2 for more details.
//
// You should have received a copy of the GNU General Public License version 2 along with
// OpenFV. If not, see https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html.

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
#include "typedefs.h"
// #include "tools.h"
// #include "visualize.h"

#ifndef WITHOUT_CUDA
#include "cuda_lib.h"
#endif

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace std;
using namespace cv;

/*!
  Class with functions that allow user to calculate synthetic aperture refocused
  images using calibration data.
*/
class saRefocus {

 public:
    ~saRefocus() {

    }

    saRefocus();

    /* Probably not needed anymore
 saRefocus(refocusing_data refocusing_params, int frame, int mult, double mult_exp):
    P_mats_(refocusing_params.P_mats), P_mats_u_(refocusing_params.P_mats_u), cam_names_(refocusing_params.cam_names), img_size_(refocusing_params.img_size), scale_(refocusing_params.scale), num_cams_(refocusing_params.num_cams), warp_factor_(refocusing_params.warp_factor), z_(0), thresh_(0), mult_(mult), mult_exp_(mult_exp) { }
    */

    /*! saRefocus constructor
      \param settings A refocus_settings struct variable containing all relevant
      settings
    */
    saRefocus(refocus_settings settings);

    /*! saRefocus construction to be generally used when rendered data (using Scene and Camera
      classes is being used
      \param num_cams Number of cameras that will be in the simulated array
      \param Factor indicating number of physical units per pixel
    */
    saRefocus(int num_cams, double f);

    int num_cams() { return num_cams_; }
    double scale() { return scale_; }
    Size img_size() { return img_size_; }
    int num_frames() { return imgs[0].size(); }

    void set_init_z(double z) { z_ = z; }
    void set_drx(double drx) { drx_ = drx; }
    void set_dry(double dry) { dry_ = dry; }
    void set_dx(double dx) { dx_ = dx; }
    void set_dy(double dy) { dy_ = dy; }
    void set_dz(double dz) { dz_ = dz; }
    void set_undistort(int flag) { UNDISTORT_IMAGES = flag; }
    void set_perspective_shift(int flag) { PERSPECTIVE_SHIFT = flag; }

    // DocString: read_calib_data
    //! Read refractive calibration data
    void read_calib_data(string path);
    // DocString: read_kalibr_data
    //! Read calibration data that is output from kalibr
    void read_kalibr_data(string path);
    // DocString: read_calib_data_pin
    //! Read pinhole calibration data
    void read_calib_data_pin(string path);

    // DocString: read_imgs
    //! Read images when they are as individual files in separate folders
    void read_imgs(string path);
    // DocString: read_imgs_mtiff
    //! Read images when they are in multipage TIFF files
    void read_imgs_mtiff(string path);
    //! Read images when they are in mp4 files
    //! Read images when they are in mp4 files
    void read_imgs_mp4(string path);

    void CPUliveView();

    void initializeRefocus();
    void initializeCPU();

    // DocString: refocus
    /*! Calculate a refocused image
      \param z Depth in physical units at which to calculate refocused image
      \param rx Angle by which to rotate focal plane about x axis
      \param ry Angle by which you rotate focal plane about y axis
      \param rz Angle by which to rotate focal plane about z axis
      \param thresh Thresholding level (if additive refocusing is used)
      \param frame The frame (int time) to refocus. Indexing starts at 0.
      \return Refocused image as OpenCV Mat type
    */
    Mat refocus(double z, double rx, double ry, double rz, double thresh, int frame);

#ifndef WITHOUT_CUDA
    // DocString: GPUliveView
    //! Start refocusing live view (requires Qt)
    void GPUliveView();
    // DocString: initializeGPU
    /*! Initialize everything required to calculate refocused images on GPU.
      This is automatically called by GPUliveView() but needs to be explicitly
      called when live view is not being started.
    */
    void initializeGPU();
    void GPUrefocus(int live, int frame);
    void GPUrefocus_ref(int live, int frame);
    void GPUrefocus_ref_corner(int live, int frame);
#endif

    void CPUrefocus(int live, int frame);
    void CPUrefocus_ref(int live, int frame);
    void CPUrefocus_ref_corner(int live, int frame);

    static void cb_mult(int, void*);
    static void cb_undistort(int, void*);
    static void cb_frames(int, void*);
    static void cb_dz_p1(int, void*);
    static void cb_dz_1(int, void*);
    static void cb_dz_10(int, void*);
    static void cb_dz_100(int, void*);
    static void cb_drx(int);
    static void cb_dry(int, void*);
    void updateLiveFrame();
    void liveViewWindow(Mat img);

    void dump_stack(string path, double zmin, double zmax, double dz, double thresh, string type);
    void dump_stack_piv(string path, double zmin, double zmax, double dz, double thresh, string type, int f, vector<Mat> &returnStack);
    void calculateQ(double zmin, double zmax, double dz, double thresh, int frame, string refPath);
    void return_stack(double zmin, double zmax, double dz, double thresh, int frame, vector<Mat> &stack);
    void return_stack(double zmin, double zmax, double dz, double thresh, int frame, vector<Mat> &stack, double &time);
    double getQ(vector<Mat> &stack, vector<Mat> &refStack);

    // Expert mode functions
    void setBenchmarkMode(int);
    void setIntImgMode(int);
    void setGpuDevice(int id);
    void setGpuMode(int flag);
    void setSingleCamDebug(int);
    void setStdevThresh(int);
    void setArrayData(vector<Mat> imgs, vector<Mat> Pmats, vector<Mat> cam_locations);
    void addView(Mat img, Mat P, Mat location);
    void addViews(vector< vector<Mat> > frames, vector<Mat> Ps, vector<Mat> locations);
    void clearViews();
    void setF(double f);
    void setMult(int flag, double exp);
    void setHF(int hf);
    void setRefractive(int ref, double zW, double n1, double n2, double n3, double t);
    string showSettings();

    Mat project_point(int cam, Mat_<double> X);
    Mat getP(int);
    Mat getC(int);
    vector< vector<Mat> > getCamStacks();

 protected:
    // Vector of vectors that stores images from all cameras and for all time steps
    vector< vector<Mat> > imgs;

 private:

    void calc_ref_refocus_map(Mat_<double> Xcam, double z, Mat_<double> &x, Mat_<double> &y, int cam);
    void calc_refocus_map(Mat_<double> &x, Mat_<double> &y, int cam);
    void calc_ref_refocus_H(Mat_<double> Xcam, double z, int cam, Mat &H);
    void calc_refocus_H(int cam, Mat &H);
    void img_refrac(Mat_<double> Xcam, Mat_<double> X, Mat_<double> &X_out);

    void threshold_image(Mat &refocused);
    void apply_preprocess(void (*preprocess_func)(Mat, Mat), string path);
    void adaptiveNorm(Mat in, Mat &out, int xf, int yf);
    void slidingMinToZero(Mat in, Mat &out, int xf, int yf);

#ifndef WITHOUT_CUDA
    void uploadToGPU();
    void uploadToGPU_ref();
    void threshold_image(GpuMat &refocused);
#endif

    // Refocusing result
    Mat result_;

    // data types and private functions
    vector<Mat> P_mats_;
    vector<Mat> P_mats_u_;
    Mat P_mat_avg_;
    vector<Mat> K_mats_;
    vector<Mat> R_mats_;
    vector<Mat> t_vecs_;
    vector<string> cam_names_;
    vector<Mat> cam_locations_;
    vector<Mat> dist_coeffs_;
    Size calib_img_size_;
    Size img_size_;
    double scale_;
    int num_cams_;
    int imgs_read_;
    vector<Mat> stack_;
    vector< vector<Mat> > cam_stacks_;

    // Scene geometry params
    float geom[5];

    // Refocusing parameters
    double z_, dz_, zs_, xs_, ys_, dx_, dy_, rx_, ry_, rz_, drx_, dry_, drz_, cxs_, cys_, czs_, crx_, cry_, crz_;
    double thresh_;
    vector<int> frames_;
    vector<int> shifts_;
    int mult_;
    double mult_exp_;
    int minlos_;
    double warp_factor_;
    int active_frame_;
    int start_frame_;
    int end_frame_;
    int skip_frame_;
    double rf_;

    Mat refocused_host_;

    Mat cputemp; Mat cputemp2; Mat cpurefocused;

#ifndef WITHOUT_CUDA
    vector<gpu::GpuMat> array;
    vector<gpu::GpuMat> P_mats_gpu;
    vector<gpu::GpuMat> cam_locations_gpu;
    vector<gpu::GpuMat> xmaps, ymaps;
    vector< vector<gpu::GpuMat> > array_all;
    gpu::GpuMat temp, temp2, refocused, xmap, ymap;
#endif

    int frame_to_upload_;

    int GPU_FLAG;
    int REF_FLAG;
    int CORNER_FLAG; // Flag to use corner based homography fit method
    int MTIFF_FLAG;
    int MP4_FLAG;
    int ALL_FRAME_FLAG;
    int INVERT_Y_FLAG;
    int EXPERT_FLAG;
    // TODO: Enable setting this flag outside expert mode. Will need
    // to add a variable to the refocus_settings struct.
    int STDEV_THRESH;
    int SINGLE_CAM_DEBUG;
    double IMG_REFRAC_TOL;
    int MAX_NR_ITERS;
    int BENCHMARK_MODE;
    int INT_IMG_MODE;
    int RESIZE_IMAGES;
    int UNDISTORT_IMAGES;
    int KALIBR;
    int PERSPECTIVE_SHIFT;

};

#endif
