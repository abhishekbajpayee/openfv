//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//                           License Agreement
//                For Open Source Flow Visualization Library
//
// Copyright 2013-2015 Abhishek Bajpayee
//
// This file is part of openFV.
//
// openFV is free software: you can redistribute it and/or modify it under the terms of the 
// GNU General Public License as published by the Free Software Foundation, either version 
// 3 of the License, or (at your option) any later version.
//
// openFV is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
// See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with openFV. 
// If not, see http://www.gnu.org/licenses/.

// -------------------------------------------------------
// -------------------------------------------------------
// Synthetic Aperture - Particle Tracking Velocimetry Code
// --- Refocusing Library ---
// -------------------------------------------------------
// Author: Abhishek Bajpayee
//         Dept. of Mechanical Engineering
//         Massachusetts Institute of Technology
// -------------------------------------------------------
// -------------------------------------------------------

#include "std_include.h"
#include "calibration.h"
#include "refocusing.h"
#include "tools.h"
#include "cuda_lib.h"
#include "visualize.h"

#include <Eigen/Core>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace std;
using namespace cv;

saRefocus::saRefocus() {

    GPU_FLAG=1;
    REF_FLAG=0;
    CORNER_FLAG=0;
    MTIFF_FLAG=0;
    INVERT_Y_FLAG=0;
    EXPERT_FLAG=1;
    frame_=-1;
    mult_=0;
    preprocess_=0;

    frames_.push_back(0);

    num_cams_ = 0;

}

saRefocus::saRefocus(int num_cams, double f) {

    LOG(INFO)<<"Refocusing object created in expert mode";
    LOG(INFO)<<"Note: requires manual tweaking of parameters!";

    GPU_FLAG=1;
    REF_FLAG=0;
    CORNER_FLAG=0;
    MTIFF_FLAG=0;
    INVERT_Y_FLAG=0;
    EXPERT_FLAG=1;
    frame_=-1;
    mult_=0;
    preprocess_=0;
    
    frames_.push_back(0);

    num_cams_ = num_cams;

    scale_ = f;

}

saRefocus::saRefocus(refocus_settings settings):
    GPU_FLAG(settings.gpu), CORNER_FLAG(settings.hf_method), MTIFF_FLAG(settings.mtiff), frame_(settings.upload_frame), mult_(settings.mult), preprocess_(settings.preprocess) {

    read_calib_data(settings.calib_file_path);
    
    //} else {
    //read_calib_data_pin(settings.calib_file_path);
    //}

    if (mult_) {
        mult_exp_ = settings.mult_exp;
    }

    if (preprocess_) {
        parse_preprocess_settings(settings.preprocess_file);
    }

    if (MTIFF_FLAG) {
        vector<int> frames;
        if (settings.all_frames) {
            ALL_FRAME_FLAG = 1;
        } else {
            ALL_FRAME_FLAG = 0;
            int begin = settings.start_frame;
            int end = settings.end_frame;
            for (int i=begin; i<=end; i++)
                frames_.push_back(i);
        }
        read_imgs_mtiff(settings.images_path);
    } else {
        read_imgs(settings.images_path);
    }

    z_ = 0; 
    xs_ = 0; ys_ = 0; zs_ = 0; 
    rx_ = 0; ry_ = 0; rz_ = 0;
    cxs_ = 0; cys_ = 0; czs_ = 0;
    crx_ = 0; cry_ = 0; crz_ = 0;

}

void saRefocus::read_calib_data(string path) {
 
    ifstream file;

    file.open(path.c_str());
    if(file.fail())
        LOG(FATAL)<<"Could not open calibration file! Terminating..."<<endl;

    LOG(INFO)<<"LOADING REFRACTIVE CALIBRATION DATA...";

    string time_stamp;
    getline(file, time_stamp);
    VLOG(3)<<time_stamp;

    double avg_reproj_error_;
    file>>avg_reproj_error_;

    file>>img_size_.width;
    file>>img_size_.height;
    file>>scale_;

    file>>num_cams_;

    string cam_name;

    for (int n=0; n<num_cams_; n++) {
        
        for (int i=0; i<2; i++) getline(file, cam_name);
        VLOG(3)<<"cam_names_["<<n<<"] = "<<cam_name<<endl;
        cam_names_.push_back(cam_name);

        Mat_<double> P_mat = Mat_<double>::zeros(3,4);
        for (int i=0; i<3; i++) {
            for (int j=0; j<4; j++) {
                file>>P_mat(i,j);
            }
        }
        P_mats_.push_back(P_mat);
        VLOG(3)<<"P_mat["<<n<<"]"<<endl<<P_mat<<endl;

        Mat_<double> loc = Mat_<double>::zeros(3,1);
        for (int i=0; i<3; i++)
            file>>loc(i,0);

        VLOG(3)<<"cam_locations_["<<n<<"]"<<endl<<loc<<endl;
        cam_locations_.push_back(loc);

    }

    file>>REF_FLAG;
    if (REF_FLAG) {
        LOG(INFO)<<"Calibration is refractive";
        file>>geom[0]; file>>geom[4]; file>>geom[1]; file>>geom[2]; file>>geom[3];
    } else {
        LOG(INFO)<<"Calibration is pinhole";
    }

    LOG(INFO)<<"DONE"<<endl;

}

void saRefocus::read_calib_data_pin(string path) {

    ifstream file;
    file.open(path.c_str());
    if(file.fail())
        LOG(FATAL)<<"Could not open calibration file! Termintation..."<<endl;

    LOG(INFO)<<"LOADING PINHOLE CALIBRATION DATA...";

    string time_stamp;
    getline(file, time_stamp);

    double reproj_error1, reproj_error2;
    file>>reproj_error1>>reproj_error2;
    file>>num_cams_;

    Mat_<double> P_u = Mat_<double>::zeros(3,4);
    Mat_<double> P = Mat_<double>::zeros(3,4);
    string cam_name;
    double tmp;
    
    for (int i=0; i<num_cams_; i++) {
        
        for (int j=0; j<2; j++) getline(file, cam_name);
        cam_names_.push_back(cam_name);
        
        for (int j=0; j<3; j++) {
            for (int k=0; k<3; k++) {
                file>>P_u(j,k);
            }
            file>>P_u(j,3);
        }
        for (int j=0; j<3; j++) {
            for (int k=0; k<3; k++) {
                file>>P(j,k);
            }
            file>>P(j,3);
        }
        //refocusing_params_.P_mats_u.push_back(P_u.clone());
        P_mats_.push_back(P.clone());

    }

    file>>img_size_.width;
    file>>img_size_.height;
    file>>scale_;
    file>>warp_factor_;

    file.close();

    LOG(INFO)<<"DONE"<<endl;

}

void saRefocus::read_imgs(string path) {

    DIR *dir;
    struct dirent *ent;
 
    string dir1(".");
    string dir2("..");
    string temp_name;
    string img_prefix = "";

    Mat image, fimage;

    vector<string> img_names;

    LOG(INFO)<<"READING IMAGES TO REFOCUS...";
    VLOG(1)<<"\n";

    for (int i=0; i<num_cams_; i++) {

        VLOG(1)<<"Camera "<<i+1<<" of "<<num_cams_<<"..."<<endl;

        string path_tmp;
        vector<Mat> refocusing_imgs_sub;

        path_tmp = path+cam_names_[i]+"/"+img_prefix;
        
        dir = opendir(path_tmp.c_str());
        while(ent = readdir(dir)) {
            temp_name = ent->d_name;
            if (temp_name.compare(dir1)) {
                if (temp_name.compare(dir2)) {
                    string path_img = path_tmp+temp_name;
                    img_names.push_back(path_img);
                }
            }
        }

        sort(img_names.begin(), img_names.end());
        for (int j=0; j<img_names.size(); j++) {
            VLOG(1)<<j<<": "<<img_names[j]<<endl;
            image = imread(img_names[j], 0);
            // image = imread(img_names[i]);
            Mat imgI;
            // preprocess(image, imgI);
            //refocusing_imgs_sub.push_back(imgI.clone());
            refocusing_imgs_sub.push_back(image.clone());
            if (i==0) {
                frames_.push_back(j);
            }
        }
        img_names.clear();

        imgs.push_back(refocusing_imgs_sub);
        path_tmp = "";

        VLOG(1)<<"done!\n";
   
    }
 
    VLOG(3)<<"Converting image types to 32 bit float...";
    initializeRefocus();

    LOG(INFO)<<"DONE READING IMAGES"<<endl;

}

void saRefocus::read_imgs_mtiff(string path) {
    
    LOG(INFO)<<"READING IMAGES TO REFOCUS...";
    VLOG(1)<<"\n";

    DIR *dir;
    struct dirent *ent;

    string dir1(".");
    string dir2("..");
    string temp_name;

    vector<string> img_names;

    dir = opendir(path.c_str());
    while(ent = readdir(dir)) {
        temp_name = ent->d_name;
        if (temp_name.compare(dir1)) {
            if (temp_name.compare(dir2)) {
                if (temp_name.compare(temp_name.size()-3,3,"tif") == 0) {
                    string img_name = path+temp_name;
                    img_names.push_back(img_name);
                }
            }
        }
    }

    sort(img_names.begin(), img_names.end());
    vector<TIFF*> tiffs;

    VLOG(1)<<"Images in path:"<<endl;
    for (int i=0; i<img_names.size(); i++) {
        VLOG(1)<<img_names[i]<<endl;
        TIFF* tiff = TIFFOpen(img_names[i].c_str(), "r");
        tiffs.push_back(tiff);
    }

    VLOG(1)<<"Counting number of frames...";
    int dircount = 0;
    if (tiffs[0]) {
	do {
	    dircount++;
	} while (TIFFReadDirectory(tiffs[0]));
    }
    VLOG(1)<<"done! ("<<dircount<<" frames found.)"<<endl<<endl;

    if (ALL_FRAME_FLAG) {
        VLOG(1)<<"READING ALL FRAMES..."<<endl;
        for (int i=0; i<dircount; i++)
            frames_.push_back(i);
    }

    VLOG(1)<<"Reading images..."<<endl;
    for (int n=0; n<img_names.size(); n++) {
        
        VLOG(1)<<"Camera "<<n+1<<"...";

        vector<Mat> refocusing_imgs_sub;

        int frame=0; // TODO: check if this is being used at all
        int count=0;
        int skip=1400;
        
        for (int f=0; f<frames_.size(); f++) {

            Mat img, img2;
            uint32 c, r;
            size_t npixels;
            uint32* raster;
            
            TIFFSetDirectory(tiffs[n], frames_[f]);

            TIFFGetField(tiffs[n], TIFFTAG_IMAGEWIDTH, &c);
            TIFFGetField(tiffs[n], TIFFTAG_IMAGELENGTH, &r);
            npixels = r * c;
            raster = (uint32*) _TIFFmalloc(npixels * sizeof (uint32));
            if (raster != NULL) {
                if (TIFFReadRGBAImageOriented(tiffs[n], c, r, raster, ORIENTATION_TOPLEFT, 0)) {
                    img.create(r, c, CV_32F);
                    for (int i=0; i<r; i++) {
                        for (int j=0; j<c; j++) {
                            img.at<float>(i,j) = TIFFGetR(raster[i*c+j]);
                        }
                    }
                }
                _TIFFfree(raster);
            }
            
            // img.convertTo(img2, CV_8U);
            Mat imgI;
            // preprocess(img2, imgI);
            
            // imshow("img to push", imgI); waitKey(0);
            refocusing_imgs_sub.push_back(img.clone());
            count++;
            
            frame += skip;

        }

        imgs.push_back(refocusing_imgs_sub);
        VLOG(1)<<"done! "<<count<<" frames read."<<endl;

    }

    VLOG(3)<<"Converting image types to 32 bit float...";
    initializeRefocus();

    LOG(INFO)<<"DONE READING IMAGES"<<endl;

}

void saRefocus::GPUliveView() {

    initializeGPU();

    if (REF_FLAG) {
        if (CORNER_FLAG) {
            LOG(INFO)<<"Using corner based homography fit method..."<<endl;
        } else {
            LOG(INFO)<<"Using full refractive calculation method..."<<endl;
        }
    } else {
        LOG(INFO)<<"Using pinhole refocusing..."<<endl;
    }

    active_frame_ = 0; thresh = 0;

    namedWindow("Result", CV_WINDOW_AUTOSIZE);

    if (REF_FLAG) {
        if (CORNER_FLAG) {
            GPUrefocus_ref_corner(thresh, 1, active_frame_);
        } else {
            GPUrefocus_ref(thresh, 1, active_frame_);
        }
    } else {
        GPUrefocus(thresh, 1, active_frame_);
    }
    
    double dz = 0.1;
    double dthresh = 5/255.0;
    double tlimit = 1.0;
    double mult_exp_limit = 1.0;
    double mult_thresh = 0.01;

    while( 1 ){
        int key = cvWaitKey(10);
        VLOG(3)<<"Key press: "<<(key & 255)<<endl;

        if ( (key & 255)!=255 ) {

            if ( (key & 255)==83 ) {
                z_ += dz;
            } else if( (key & 255)==81 ) {
                z_ -= dz;
            } else if( (key & 255)==82 ) {
                if (mult_) {
                    if (mult_exp_<mult_exp_limit)
                        mult_exp_ += mult_thresh;
                } else {
                    if (thresh<tlimit)
                        thresh += dthresh; 
                }
            } else if( (key & 255)==84 ) {
                if (mult_) {
                    if (mult_exp_>0)
                        mult_exp_ -= mult_thresh;
                } else {
                    if (thresh>0)
                        thresh -= dthresh; 
                }
            } else if( (key & 255)==46 ) {
                if (active_frame_<array_all.size()) { 
                    active_frame_++; 
                }
            } else if( (key & 255)==44 ) {
                if (active_frame_<array_all.size()) { 
                    active_frame_--; 
                }
            } else if( (key & 255)==119 ) { // w
                rx_ += 1;
            } else if( (key & 255)==113 ) { // q
                rx_ -= 1;
            } else if( (key & 255)==115 ) { // s
                ry_ += 1;
            } else if( (key & 255)==97 ) {  // a
                ry_ -= 1;
            } else if( (key & 255)==120 ) { // x
                rz_ += 1;
            } else if( (key & 255)==122 ) { // z
                rz_ -= 1;
            } else if( (key & 255)==114 ) { // r
                xs_ += 1;
            } else if( (key & 255)==101 ) { // e
                xs_ -= 1;
            } else if( (key & 255)==102 ) { // f
                ys_ += 1;
            } else if( (key & 255)==100 ) { // d
                ys_ -= 1;
            } else if( (key & 255)==118 ) { // v
                zs_ += 1;
            } else if( (key & 255)==99 ) {  // c
                zs_ -= 1;
            } else if( (key & 255)==117 ) { // u
                crx_ += 1;
            } else if( (key & 255)==121 ) { // y
                crx_ -= 1;
            } else if( (key & 255)==106 ) { // j
                cry_ += 1;
            } else if( (key & 255)==104 ) { // h
                cry_ -= 1;
            } else if( (key & 255)==109 ) { // m
                crz_ += 1;
            } else if( (key & 255)==110 ) { // n
                crz_ -= 1;
            } else if( (key & 255)==32 ) {
                mult_ = (mult_+1)%2;
            } else if( (key & 255)==27 ) {  // ESC
                cvDestroyAllWindows();
                break;
            }
            
            // Call refocus function
            if(REF_FLAG) {
                if (CORNER_FLAG) {
                    GPUrefocus_ref_corner(thresh, 1, active_frame_);
                } else {
                    GPUrefocus_ref(thresh, 1, active_frame_);
                }
            } else {
                GPUrefocus(thresh, 1, active_frame_);
            }

        }

    }

}

void saRefocus::CPUliveView() {

    initializeCPU();

    if (CORNER_FLAG) {
        LOG(INFO)<<"Using corner based homography fit method..."<<endl;
    } else {
        LOG(INFO)<<"Using full refractive calculation method..."<<endl;
    }

    active_frame_ = 0;

    namedWindow("Result", CV_WINDOW_AUTOSIZE);
    if (REF_FLAG) {
        if (CORNER_FLAG) {
            CPUrefocus_ref_corner(z_, thresh, 1, active_frame_);
        } else {
            CPUrefocus_ref(z_, thresh, 1, active_frame_);
        }
    } else {
        CPUrefocus(z_, thresh, 1, active_frame_);
    }
    
    double dz = 0.5;
    double dthresh = 5/255.0;
    double tlimit = 1.0;

    while( 1 ){
        int key = cvWaitKey(10);
        VLOG(3)<<"Key press: "<<(key & 255)<<endl;
        
        if ( (key & 255)!=255 ) {

            if ( (key & 255)==83 ) {
                z_ += dz;
            } else if( (key & 255)==81 ) {
                z_ -= dz;
            } else if( (key & 255)==82 ) {
                if (thresh<tlimit) { 
                    thresh += dthresh; 
                }
            } else if( (key & 255)==84 ) {
                if (thresh>0) { 
                    thresh -= dthresh; 
                }
            } else if( (key & 255)==46 ) {
                if (active_frame_<array_all.size()) { 
                    active_frame_++; 
                }
            } else if( (key & 255)==44 ) {
                if (active_frame_<array_all.size()) { 
                    active_frame_--; 
                }
            } else if( (key & 255)==27 ) {
                break;
            }
            
            // Call refocus function
            if(REF_FLAG) {
                if (CORNER_FLAG) {
                    CPUrefocus_ref_corner(z_, thresh, 1, active_frame_);
                } else {
                    CPUrefocus_ref(z_, thresh, 1, active_frame_);
                }
            } else {
                CPUrefocus(z_, thresh, 1, active_frame_);
            }

        }

    }

}

Mat saRefocus::refocus(double z, double rx, double ry, double rz, double thresh, int frame) {

    z_ = z;
    rx_ = rx;
    ry_ = ry;
    rz_ = rz;
    thresh /= 255.0;

    if (REF_FLAG) {
        if (CORNER_FLAG) {
            if (GPU_FLAG) {
                GPUrefocus_ref_corner(thresh, 0, frame);
            } else {
                CPUrefocus_ref_corner(z_, thresh, 0, frame);
            }
        } else {
            if (GPU_FLAG) {
                GPUrefocus_ref(thresh, 0, frame);
            } else {
                CPUrefocus_ref(z_, thresh, 0, frame);
            }
        }
    } else {
        if (GPU_FLAG) {
            GPUrefocus(thresh, 0, frame);
        } else {
            CPUrefocus(z_, thresh, 0, frame);
        }
    }

    return(result_);

}

void saRefocus::initializeRefocus() {

    // This functions converts any incoming datatype images to
    // CV_32F ranging between 0 and 1
    // Note this assumes that black and white pixel value depends
    // on the datatype
    // TODO: add ability to handle more data types

    int type = imgs[0][0].type();

    for (int i=0; i<imgs.size(); i++) {
        for (int j=0; j<imgs[i].size(); j++) {

            Mat img;
            switch(type) {

            case CV_8U:
                imgs[i][j].convertTo(img, CV_32F);
                img /= 255.0;
                imgs[i][j] = img.clone();
                break;

            case CV_16U:
                imgs[i][j].convertTo(img, CV_32F);
                img /= 65535.0;
                imgs[i][j] = img.clone();
                break;

            case CV_32F:
                break;

            case CV_64F:
                imgs[i][j].convertTo(img, CV_32F);
                imgs[i][j] = img.clone();
                break;

            }


        }
    }

    //preprocess();

}

// TODO: This function prints free memory on GPU and then
//       calls uploadToGPU() which uploads either a given
//       frame or all frames to GPU depending on frame_
void saRefocus::initializeGPU() {
    
    if (!EXPERT_FLAG) {

        LOG(INFO)<<endl<<"INITIALIZING GPU..."<<endl;

        LOG(INFO)<<"CUDA Enabled GPU Devices: "<<gpu::getCudaEnabledDeviceCount<<endl;
    
        gpu::DeviceInfo gpuDevice(gpu::getDevice());
    
        LOG(INFO)<<"---"<<gpuDevice.name()<<"---"<<endl;
        LOG(INFO)<<"Total Memory: "<<(gpuDevice.totalMemory()/pow(1024.0,2))<<" MB"<<endl;
    }

    uploadToGPU();

    if (REF_FLAG)
        if (!CORNER_FLAG)
            uploadToGPU_ref();

}

void saRefocus::initializeCPU() {

    // stuff

}

// TODO: Right now this function just starts uploading images
//       without checking if there is enough free memory on GPU
//       or not.
void saRefocus::uploadToGPU() {

    if (!EXPERT_FLAG) {
        gpu::DeviceInfo gpuDevice(gpu::getDevice());
        double free_mem_GPU = gpuDevice.freeMemory()/pow(1024.0,2);
        LOG(INFO)<<"Free Memory before: "<<free_mem_GPU<<" MB"<<endl;
    }

    if (frame_>=0) {

        VLOG(1)<<"Uploading frame "<<frame_<<" to GPU..."<<endl;
        for (int i=0; i<num_cams_; i++) {
            temp.upload(imgs[i][frame_]);
            array.push_back(temp.clone());
        }
        array_all.push_back(array);

    } else if (frame_==-1) {
        
        VLOG(1)<<"Uploading all frames to GPU..."<<endl;
        for (int i=0; i<imgs[0].size(); i++) {
            for (int j=0; j<num_cams_; j++) {
                temp.upload(imgs[j][i]);
                array.push_back(temp.clone());
            }
            array_all.push_back(array);
            array.clear();
        }
        
    } else {
        LOG(FATAL)<<"Invalid frame value to visualize!"<<endl;
    }

    if (!EXPERT_FLAG) {
        gpu::DeviceInfo gpuDevice(gpu::getDevice());
        LOG(INFO)<<"Free Memory after: "<<(gpuDevice.freeMemory()/pow(1024.0,2))<<" MB"<<endl<<endl;
    }

}

void saRefocus::uploadToGPU_ref() {

    LOG(INFO)<<"Uploading data required by full refocusing method to GPU...";

    Mat_<float> D = Mat_<float>::zeros(3,3);
    D(0,0) = scale_; D(1,1) = scale_;
    D(0,2) = img_size_.width*0.5; D(1,2) = img_size_.height*0.5;
    D(2,2) = 1;
    Mat Dinv = D.inv();
    
    float hinv[6];
    hinv[0] = Dinv.at<float>(0,0); hinv[1] = Dinv.at<float>(0,1); hinv[2] = Dinv.at<float>(0,2);
    hinv[3] = Dinv.at<float>(1,0); hinv[4] = Dinv.at<float>(1,1); hinv[5] = Dinv.at<float>(1,2);

    float locations[9][3];
    float pmats[9][12];
    for (int i=0; i<9; i++) {
        for (int j=0; j<3; j++) {
            locations[i][j] = cam_locations_[i].at<double>(j,0);
            for (int k=0; k<4; k++) {
                pmats[i][j*4+k] = P_mats_[i].at<double>(j,k);
            }
        }
    }
    
    uploadRefractiveData(hinv, locations, pmats, geom);

    Mat blank(img_size_.height, img_size_.width, CV_32F, float(0));
    xmap.upload(blank); ymap.upload(blank);
    temp.upload(blank); temp2.upload(blank); 
    //refocused.upload(blank);

    for (int i=0; i<9; i++) {
        xmaps.push_back(xmap.clone());
        ymaps.push_back(ymap.clone());
    }

    LOG(INFO)<<"done!"<<endl;

}

// ---GPU Refocusing Functions Begin--- //

void saRefocus::GPUrefocus(double thresh, int live, int frame) {

    int curve = 0;

    Mat_<double> x = Mat_<double>::zeros(img_size_.height, img_size_.width);
    Mat_<double> y = Mat_<double>::zeros(img_size_.height, img_size_.width);

    Mat xm, ym;

    // add warp factor stuff

    Scalar fact = Scalar(1/double(array_all[frame].size()));

    Mat H, trans;

    if (curve) {
        calc_refocus_map(x, y, 0); x.convertTo(xm, CV_32FC1); y.convertTo(ym, CV_32FC1); xmap.upload(xm); ymap.upload(ym);
        gpu::remap(array_all[frame][0], temp, xmap, ymap, INTER_LINEAR);
    } else {
        //T_from_P(P_mats_[0], H, z_, scale_, img_size_);
        calc_refocus_H(0, H);
        gpu::warpPerspective(array_all[frame][0], temp, H, img_size_);
    }

    if (mult_) {
        gpu::pow(temp, mult_exp_, temp2);
    } else {
        gpu::multiply(temp, fact, temp2);
    }

    refocused = temp2.clone();

    for (int i=1; i<num_cams_; i++) {
        
        if (curve) {
            calc_refocus_map(x, y, i); x.convertTo(xm, CV_32FC1); y.convertTo(ym, CV_32FC1); xmap.upload(xm); ymap.upload(ym);
            gpu::remap(array_all[frame][i], temp, xmap, ymap, INTER_LINEAR);
        } else {       
            //T_from_P(P_mats_[i], H, z, scale_, img_size_);
            calc_refocus_H(i, H);       
            gpu::warpPerspective(array_all[frame][i], temp, H, img_size_);
        }

        if (mult_) {
            gpu::pow(temp, mult_exp_, temp2);
            gpu::multiply(refocused, temp2, refocused);
        } else {
            gpu::multiply(temp, fact, temp2);
            gpu::add(refocused, temp2, refocused);        
        }

    }
    
    gpu::threshold(refocused, refocused, thresh, 0, THRESH_TOZERO);

    Mat refocused_host_(refocused);
    
    if (live) {

        char title[200];
        sprintf(title, "mult = %d, exp = %f, T = %f, frame = %d, xs = %f, ys = %f, zs = %f \nrx = %f, ry = %f, rz = %f, crx = %f, cry = %f, crz = %f", mult_, mult_exp_, thresh*255.0, frame, xs_, ys_, z_, rx_, ry_, rz_, crx_, cry_, crz_);

        imshow("Result", refocused_host_);
        displayOverlay("Result", title);

    }

    //refocused_host_.convertTo(result, CV_8U);
    result_ = refocused_host_.clone();

}

void saRefocus::GPUrefocus_ref(double thresh, int live, int frame) {

    Scalar fact = Scalar(1/double(num_cams_));
    //Mat blank(img_size_.height, img_size_.width, CV_8UC1, Scalar(0));
    Mat blank(img_size_.height, img_size_.width, CV_32F, Scalar(0));
    refocused.upload(blank);
    
    for (int i=0; i<num_cams_; i++) {

        gpu_calc_refocus_map(xmap, ymap, z_, i);
        gpu::remap(array_all[frame][i], temp, xmap, ymap, INTER_LINEAR);
        
        if (i==0) {
            Mat M; 
            xmap.download(M); writeMat(M, "../temp/xmap.txt");
            ymap.download(M); writeMat(M, "../temp/ymap.txt");
        }

        gpu::multiply(temp, fact, temp2);
        gpu::add(refocused, temp2, refocused);
        
    }
    
    gpu::threshold(refocused, refocused, thresh, 0, THRESH_TOZERO);

    refocused.download(refocused_host_);
    
    if (live) {
        char title[50];
        sprintf(title, "z = %f, thresh = %f, frame = %d", z_, thresh*255.0, frame);
        putText(refocused_host_, title, Point(10,20), FONT_HERSHEY_PLAIN, 1.0, Scalar(255,0,0));
        imshow("Result", refocused_host_);
    }
    
    result_ = refocused_host_.clone();

}

void saRefocus::GPUrefocus_ref_corner(double thresh, int live, int frame) {

    Scalar fact = Scalar(1/double(num_cams_));
    // Mat blank(img_size_.height, img_size_.width, CV_8UC1, Scalar(0));
    Mat blank(img_size_.height, img_size_.width, CV_32F, Scalar(0));
    refocused.upload(blank);
    
    Mat H;
    calc_ref_refocus_H(cam_locations_[0], z_, 0, H);
    writeMat(H, "../temp/Hcust.txt");
    gpu::warpPerspective(array_all[frame][0], temp, H, img_size_);
    

    if (mult_) {
        gpu::pow(temp, mult_exp_, temp2);
    } else {
        gpu::multiply(temp, fact, temp2);
    }

    refocused = temp2.clone();

    for (int i=1; i<num_cams_; i++) {

        calc_ref_refocus_H(cam_locations_[i], z_, i, H);
        gpu::warpPerspective(array_all[frame][i], temp, H, img_size_);

        if (mult_) {
            gpu::pow(temp, mult_exp_, temp2);
            gpu::multiply(refocused, temp2, refocused);
        } else {
            gpu::multiply(temp, fact, temp2);
            gpu::add(refocused, temp2, refocused);
        }

    }

    gpu::threshold(refocused, refocused, thresh, 0, THRESH_TOZERO);

    refocused.download(refocused_host_);

    //imwrite("../temp/refocused1.jpg", refocused_host_);

    if (live) {
        
        char title[150];
        sprintf(title, "mult = %d, exp = %f, T = %f, frame = %d, z = %f, xs = %f, ys = %f, zs = %f, rx = %f, ry = %f. rz = %f", mult_, mult_exp_, thresh*255.0, frame, z_, xs_, ys_, zs_, rx_, ry_, rz_);

        imshow("Result", refocused_host_);
        displayOverlay("Result", title);

    }

    result_ = refocused_host_.clone();

}

// ---GPU Refocusing Functions End--- //

// ---CPU Refocusing Functions Begin--- //

void saRefocus::CPUrefocus(double z, double thresh, int live, int frame) {

    z *= warp_factor_;

    Scalar fact = Scalar(1/double(imgs.size()));

    Mat H, trans;
    //T_from_P(P_mats_[0], H, z, scale_, img_size_);
    calc_refocus_H(0, H);
    warpPerspective(imgs[0][frame], cputemp, H, img_size_);

    if (mult_) {
        pow(cputemp, mult_exp_, cputemp2);
    } else {
        multiply(cputemp, fact, cputemp2);
    }

    cpurefocused = cputemp2.clone();

    for (int i=1; i<num_cams_; i++) {
        
        //T_from_P(P_mats_[i], H, z, scale_, img_size_);
        calc_refocus_H(i, H);
        warpPerspective(imgs[i][frame], cputemp, H, img_size_);

        if (mult_) {
            pow(cputemp, mult_exp_, cputemp2);
            multiply(cpurefocused, cputemp2, cpurefocused);
        } else {
            multiply(cputemp, fact, cputemp2);
            add(cpurefocused, cputemp2, cpurefocused);        
        }
    }
    
    threshold(cpurefocused, cpurefocused, thresh, 0, THRESH_TOZERO);

    Mat refocused_host_(cpurefocused);

    if (live) {
        char title[50];
        sprintf(title, "z = %f, thresh = %f, frame = %d", z/warp_factor_, thresh, frame);
        putText(refocused_host_, title, Point(10,20), FONT_HERSHEY_PLAIN, 1.0, Scalar(255,0,0));
        //line(refocused_host_, Point(646,482-5), Point(646,482+5), Scalar(255,0,0));
        //line(refocused_host_, Point(646-5,482), Point(646+5,482), Scalar(255,0,0));
        imshow("Result", refocused_host_);
    }

    //refocused_host_.convertTo(result_, CV_8U);
    result_ = refocused_host_.clone();

}

void saRefocus::CPUrefocus_ref(double z, double thresh, int live, int frame) {

    Mat_<double> x = Mat_<double>::zeros(img_size_.height, img_size_.width);
    Mat_<double> y = Mat_<double>::zeros(img_size_.height, img_size_.width);
    calc_ref_refocus_map(cam_locations_[0], z, x, y, 0);

    Mat res, xmap, ymap;
    x.convertTo(xmap, CV_32FC1);
    y.convertTo(ymap, CV_32FC1);
    remap(imgs[0][frame], res, xmap, ymap, INTER_LINEAR);

    refocused_host_ = res.clone()/9.0;
    
    for (int i=1; i<num_cams_; i++) {

        calc_ref_refocus_map(cam_locations_[i], z, x, y, i);
        x.convertTo(xmap, CV_32FC1);
        y.convertTo(ymap, CV_32FC1);

        remap(imgs[i][frame], res, xmap, ymap, INTER_LINEAR);

        refocused_host_ += res.clone()/9.0;
        
    }

    if (live) {
        char title[50];
        sprintf(title, "z = %f, thresh = %f, frame = %d", z, thresh, frame);
        putText(refocused_host_, title, Point(10,20), FONT_HERSHEY_PLAIN, 1.0, Scalar(255,0,0));
        imshow("Result", refocused_host_);
    }

    //refocused_host_.convertTo(result_, CV_8U);
    result_ = refocused_host_.clone();

}

void saRefocus::CPUrefocus_ref_corner(double z, double thresh, int live, int frame) {

    Mat H;
    calc_ref_refocus_H(cam_locations_[0], z, 0, H);

    Mat res;
    warpPerspective(imgs[0][frame], res, H, img_size_);
    refocused_host_ = res.clone()/9.0;
    
    for (int i=1; i<num_cams_; i++) {

        calc_ref_refocus_H(cam_locations_[i], z, i, H);
        warpPerspective(imgs[i][frame], res, H, img_size_);
        refocused_host_ += res.clone()/9.0;
        
    }

    if (live) {
        char title[50];
        sprintf(title, "z = %f, thresh = %f, frame = %d", z, thresh, frame);
        putText(refocused_host_, title, Point(10,20), FONT_HERSHEY_PLAIN, 1.0, Scalar(255,0,0));
        imshow("Result", refocused_host_);
    }

    //refocused_host_.convertTo(result_, CV_8U);
    result_ = refocused_host_.clone();

}

// ---CPU Refocusing Functions End--- //

void saRefocus::calc_ref_refocus_map(Mat_<double> Xcam, double z, Mat_<double> &x, Mat_<double> &y, int cam) {

    int width = img_size_.width;
    int height = img_size_.height;

    Mat_<double> D = Mat_<double>::zeros(3,3);
    D(0,0) = scale_; D(1,1) = scale_;
    D(0,2) = width*0.5;
    D(1,2) = height*0.5;
    D(2,2) = 1;
    Mat hinv = D.inv();

    Mat_<double> X = Mat_<double>::zeros(3, height*width);
    for (int i=0; i<width; i++) {
        for (int j=0; j<height; j++) {
            X(0,i*height+j) = i;
            X(1,i*height+j) = j;
            X(2,i*height+j) = 1;
        }
    }
    X = hinv*X;

    for (int i=0; i<X.cols; i++)
        X(2,i) = z;

    //cout<<"Refracting points"<<endl;
    Mat_<double> X_out = Mat_<double>::zeros(4, height*width);
    img_refrac(Xcam, X, X_out);

    //cout<<"Projecting to find final map"<<endl;
    Mat_<double> proj = P_mats_[cam]*X_out;
    for (int i=0; i<width; i++) {
        for (int j=0; j<height; j++) {
            int ind = i*height+j; // TODO: check this indexing
            proj(0,ind) /= proj(2,ind);
            proj(1,ind) /= proj(2,ind);
            x(j,i) = proj(0,ind);
            y(j,i) = proj(1,ind);
        }
    }

}

void saRefocus::calc_refocus_map(Mat_<double> &x, Mat_<double> &y, int cam) {

    int width = img_size_.width;
    int height = img_size_.height;

    Mat_<double> D = Mat_<double>::zeros(3,3);
    D(0,0) = scale_; D(1,1) = scale_;
    D(0,2) = width*0.5;
    D(1,2) = height*0.5;
    D(2,2) = 1;
    Mat hinv = D.inv();

    Mat_<double> X = Mat_<double>::zeros(3, height*width);
    for (int i=0; i<width; i++) {
        for (int j=0; j<height; j++) {
            X(0,i*height+j) = i;
            X(1,i*height+j) = j;
            X(2,i*height+j) = 1;
        }
    }
    X = hinv*X;

    double r = 50;
    r = r*warp_factor_;
    Mat_<double> X2 = Mat_<double>::zeros(4, height*width);
    for (int j=0; j<X.cols; j++) {
        X2(0,j) = X(0,j);
        X2(1,j) = X(1,j);
        X2(2,j) = r - r*cos(asin(X(0,j)/r)) + z_;
        X2(3,j) = 1;
    }

    //cout<<"Projecting to find final map"<<endl;
    Mat_<double> proj = P_mats_[cam]*X2;
    for (int i=0; i<width; i++) {
        for (int j=0; j<height; j++) {
            int ind = i*height+j; // TODO: check this indexing
            proj(0,ind) /= proj(2,ind);
            proj(1,ind) /= proj(2,ind);
            x(j,i) = proj(0,ind);
            y(j,i) = proj(1,ind);
        }
    }

}

void saRefocus::calc_ref_refocus_H(Mat_<double> Xcam, double z, int cam, Mat &H) {
    
    int width = img_size_.width;
    int height = img_size_.height;

    Mat_<double> D = Mat_<double>::zeros(3,3);
    D(0,0) = scale_; D(1,1) = scale_;
    D(0,2) = width*0.5;
    D(1,2) = height*0.5;
    D(2,2) = 1;
    Mat hinv = D.inv();

    Mat_<double> X = Mat_<double>::zeros(3, 4);
    X(0,0) = 0;       X(1,0) = 0;
    X(0,3) = width-1; X(1,3) = 0;
    X(0,2) = width-1; X(1,2) = height-1;
    X(0,1) = 0;       X(1,1) = height-1;
    for (int i=0; i<X.cols; i++)
        X(2,i) = 1.0;

    Mat_<double> A = X.clone();

    X = hinv*X;

    for (int i=0; i<X.cols; i++)
        X(2,i) = z;

    Mat R = getRotMat(rx_, ry_, rz_);
    X = R*X;

    //cout<<"Refracting points"<<endl;
    Mat_<double> X_out = Mat_<double>::zeros(4, 4);
    img_refrac(Xcam, X, X_out);

    //cout<<"Projecting to find final map"<<endl;
    Mat_<double> proj = P_mats_[cam]*X_out;

    Point2f src, dst;
    vector<Point2f> sp, dp;
    int i, j;

    for (int i=0; i<X.cols; i++) {
        src.x = A.at<double>(0,i); src.y = A.at<double>(1,i);
        //src.x = X(0,i); src.y = X(1,i);
        dst.x = proj(0,i)/proj(2,i); dst.y = proj(1,i)/proj(2,i);
        sp.push_back(src); dp.push_back(dst);
    }

    H = getPerspectiveTransform(dp, sp);
    //H = findHomography(dp, sp, 0);
    //H = D*H;

}

void saRefocus::calc_refocus_H(int cam, Mat &H) {

    int width = img_size_.width;
    int height = img_size_.height;

    Mat D;
    if (INVERT_Y_FLAG) {
        D = (Mat_<double>(3,3) << scale_, 0, width*0.5, 0, -1.0*scale_, height*0.5, 0, 0, 1); 
    } else {
        D = (Mat_<double>(3,3) << scale_, 0, width*0.5, 0, scale_, height*0.5, 0, 0, 1); 
    }
    Mat hinv = D.inv();

    Mat_<double> X = Mat_<double>::zeros(3, 4);
    X(0,0) = 0;       X(1,0) = 0;
    X(0,1) = width-1; X(1,1) = 0;
    X(0,2) = width-1; X(1,2) = height-1;
    X(0,3) = 0;       X(1,3) = height-1;
    X = hinv*X;

    for (int i=0; i<X.cols; i++)
        X(2,i) = 0;//z_;

    Mat R = getRotMat(rx_, ry_, rz_);
    X = R*X;

    Mat_<double> X2 = Mat_<double>::zeros(4, 4);
    for (int j=0; j<X.cols; j++) {
        X2(0,j) = X(0,j)+xs_;
        X2(1,j) = X(1,j)+ys_;
        X2(2,j) = X(2,j)+z_;
        X2(3,j) = 1;
    }

    //cout<<"Projecting to find final map"<<endl;
    Mat_<double> proj = P_mats_[cam]*X2;
    //Mat_<double> proj2 = P_mats_[4]*X2;

    /*
    Mat K, Rot, t;
    decomposeProjectionMatrix(P_mats_[4], K, Rot, t);
    Rot = getRotMat(crx_, cry_, crz_)*Rot;

    Mat_<double> P_mat_new = Mat_<double>::zeros(3,4);
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            P_mat_new(i,j) = Rot.at<double>(i,j);
        }
        P_mat_new(i,3) = -t.at<double>(i,0)/t.at<double>(3,0); // WHY NEGATIVE?
    }
    P_mat_new = K*P_mat_new;
    Mat_<double> proj2 = P_mat_new*X2;
    */

    Point2f src, dst;
    vector<Point2f> sp, dp;
    int i, j;

    for (int i=0; i<X.cols; i++) {
        src.x = X(0,i); src.y = X(1,i);
        //src.x = proj2(0,i)/proj2(2,i); src.y = proj2(1,i)/proj2(2,i);
        dst.x = proj(0,i)/proj(2,i); dst.y = proj(1,i)/proj(2,i);
        sp.push_back(src); dp.push_back(dst);
    }

    //H = findHomography(dp, sp, CV_RANSAC);
    H = findHomography(dp, sp, 0);
    H = D*H;

}

void saRefocus::img_refrac(Mat_<double> Xcam, Mat_<double> X, Mat_<double> &X_out) {

    float zW_ = geom[0]; float n1_ = geom[1]; float n2_ = geom[2]; float n3_ = geom[3]; float t_ = geom[4];

    double c[3];
    for (int i=0; i<3; i++)
        c[i] = Xcam.at<double>(0,i);

    for (int n=0; n<X.cols; n++) {

        double a[3];
        double b[3];
        double point[3];
        for (int i=0; i<3; i++)
            point[i] = X(i,n);

        a[0] = c[0] + (point[0]-c[0])*(zW_-c[2])/(point[2]-c[2]);
        a[1] = c[1] + (point[1]-c[1])*(zW_-c[2])/(point[2]-c[2]);
        a[2] = zW_;
        b[0] = c[0] + (point[0]-c[0])*(t_+zW_-c[2])/(point[2]-c[2]);
        b[1] = c[1] + (point[1]-c[1])*(t_+zW_-c[2])/(point[2]-c[2]);
        b[2] = t_+zW_;
        
        double rp = sqrt( pow(point[0]-c[0],2) + pow(point[1]-c[1],2) );
        double dp = point[2]-b[2];
        double phi = atan2(point[1]-c[1],point[0]-c[0]);

        double ra = sqrt( pow(a[0]-c[0],2) + pow(a[1]-c[1],2) );
        double rb = sqrt( pow(b[0]-c[0],2) + pow(b[1]-c[1],2) );
        double da = a[2]-c[2];
        double db = b[2]-a[2];
        
        double f, g, dfdra, dfdrb, dgdra, dgdrb;
        
        // Newton Raphson loop to solve for Snell's law
        double tol=1E-8;

        for (int i=0; i<20; i++) {

            f = ( ra/sqrt(pow(ra,2)+pow(da,2)) ) - ( (n2_/n1_)*(rb-ra)/sqrt(pow(rb-ra,2)+pow(db,2)) );
            g = ( (rb-ra)/sqrt(pow(rb-ra,2)+pow(db,2)) ) - ( (n3_/n2_)*(rp-rb)/sqrt(pow(rp-rb,2)+pow(dp,2)) );
            
            dfdra = ( (1.0)/sqrt(pow(ra,2)+pow(da,2)) )
                - ( pow(ra,2)/pow(pow(ra,2)+pow(da,2),1.5) )
                + ( (n2_/n1_)/sqrt(pow(ra-rb,2)+pow(db,2)) )
                - ( (n2_/n1_)*(ra-rb)*(2*ra-2*rb)/(2*pow(pow(ra-rb,2)+pow(db,2),1.5)) );

            dfdrb = ( (n2_/n1_)*(ra-rb)*(2*ra-2*rb)/(2*pow(pow(ra-rb,2)+pow(db,2),1.5)) )
                - ( (n2_/n1_)/sqrt(pow(ra-rb,2)+pow(db,2)) );

            dgdra = ( (ra-rb)*(2*ra-2*rb)/(2*pow(pow(ra-rb,2)+pow(db,2),1.5)) )
                - ( (1.0)/sqrt(pow(ra-rb,2)+pow(db,2)) );

            dgdrb = ( (1.0)/sqrt(pow(ra-rb,2)+pow(db,2)) )
                + ( (n3_/n2_)/sqrt(pow(rb-rp,2)+pow(dp,2)) )
                - ( (ra-rb)*(2*ra-2*rb)/(2*pow(pow(ra-rb,2)+pow(db,2),1.5)) )
                - ( (n3_/n2_)*(rb-rp)*(2*rb-2*rp)/(2*pow(pow(rb-rp,2)+pow(dp,2),1.5)) );

            ra = ra - ( (f*dgdrb - g*dfdrb)/(dfdra*dgdrb - dfdrb*dgdra) );
            rb = rb - ( (g*dfdra - f*dgdra)/(dfdra*dgdrb - dfdrb*dgdra) );

        }

        a[0] = ra*cos(phi) + c[0];
        a[1] = ra*sin(phi) + c[1];

        X_out(0,n) = a[0];
        X_out(1,n) = a[1];
        X_out(2,n) = a[2];
        X_out(3,n) = 1.0;

    }

}

void saRefocus::dump_stack(string path, double zmin, double zmax, double dz, double thresh, string type) {

    LOG(INFO)<<"SAVING STACK TO "<<path<<endl;
    
    for (int f=0; f<frames_.size(); f++) {
        
        stringstream fn;
        fn<<path<<frames_[f];
        mkdir(fn.str().c_str(), S_IRWXU);

        LOG(INFO)<<"Saving frame "<<f<<"...";

        vector<Mat> stack;
        for (double z=zmin; z<=zmax; z+=dz) {

            Mat img = refocus(z, 0, 0, 0, thresh, f);
            //Mat result = refocused_host_.clone();
            stack.push_back(img);

            // stringstream ss;
            // ss<<fn.str()<<"/"<<((z-zmin)/dz)<<"."<<type;
            // imwrite(ss.str(), img);

        }
        
        imageIO io(fn.str());
        io<<stack; stack.clear();

        LOG(INFO)<<"done!"<<endl;

    }

    LOG(INFO)<<"SAVING COMPLETE!"<<endl;

}

void saRefocus::dump_stack_piv(string path, double zmin, double zmax, double dz, double thresh, string type, int f, double q) {

    LOG(INFO)<<"SAVING STACK TO "<<path<<endl;
    
    //for (int f=0; f<frames_.size(); f++) {
        
    stringstream fn;
    fn<<path<<f;
    mkdir(fn.str().c_str(), S_IRWXU);

    string qfn = fn.str() + "/q.txt";
    fileIO qio(qfn);
    qio<<q;

    fn<<"/refocused";
    mkdir(fn.str().c_str(), S_IRWXU);
    
    LOG(INFO)<<"Saving frame "<<f<<"...";
    
    vector<Mat> stack;
    for (double z=zmin; z<=zmax; z+=dz) {
        Mat img = refocus(z, 0, 0, 0, thresh, 0);
        stack.push_back(img);
    }
        
    imageIO io(fn.str());
    io<<stack; stack.clear();

    LOG(INFO)<<"done!"<<endl;

    //}

    LOG(INFO)<<"SAVING COMPLETE!"<<endl;

}

// Function to reconstruct a volume and then compare to reference stack and calculate Q without
// dumping stack
void saRefocus::calculateQ(double zmin, double zmax, double dz, double thresh, int frame, string refPath) {

    // get refStack
    string stackPath = refPath + "stack/";
    vector<string> img_names;
    listDir(stackPath, img_names);
    sort(img_names.begin(), img_names.end());

    vector<Mat> refStack;
    LOG(INFO)<<"Reading reference stack from "<<stackPath;
    readImgStack(img_names, refStack);
    LOG(INFO)<<"done.";

    vector<Mat> stack;
    LOG(INFO)<<"Reconstructing volume...";
    return_stack(zmin, zmax, dz, thresh, frame, stack);

    LOG(INFO)<<"Calculating Q...";
    double q = getQ(stack, refStack);

    LOG(INFO)<<q;

}

void saRefocus::return_stack(double zmin, double zmax, double dz, double thresh, int frame, vector<Mat> &stack) {

    for (double z=zmin; z<=zmax+(dz*0.5); z+=dz) {
        Mat img = refocus(z, 0, 0, 0, thresh, frame);
        stack.push_back(img);
    }

}

double saRefocus::getQ(vector<Mat> &stack, vector<Mat> &refStack) {

    double xct=0;
    double xc1=0;
    double xc2=0;

    for (int i=0; i<stack.size(); i++) {

        Mat a; multiply(stack[i], refStack[i], a);
        xct += double(sum(a)[0]);

        Mat b; pow(stack[i], 2, b);
        xc1 += double(sum(b)[0]);

        Mat c; pow(refStack[i], 2, c);
        xc2 += double(sum(c)[0]);

    }

    double q = xct/sqrt(xc1*xc2);

    return(q);

}

// ---Preprocessing related functions--- //

void saRefocus::preprocess(Mat in, Mat &out) {

    if (preprocess_) {

        Mat im = in.clone();
        int tc = 0; int gbc = 0; int anc = 0; int mfc = 0; int sMeanc = 0; int smtzc = 0;

        for (int i=0; i<pp_ops.size(); i++) {
            
            Mat im2;

            switch(pp_ops[i]) {

            case 1:
                threshold(im, im2, thresh_vals[tc], 0, THRESH_TOZERO);
                VLOG(1)<<"Applied threshold at "<<thresh_vals[tc]<<endl;
                tc++;
                break;

            case 2:
                GaussianBlur(im, im2, Size(gbkernel[gbc], gbkernel[gbc]), gbsigma[gbc]);
                VLOG(1)<<"Applied gaussianBlur with kernel size "<<gbkernel[gbc]<<" and sigma "<<gbsigma[gbc]<<endl;
                gbc++;
                break;

            case 3:
                adaptiveNorm(im, im2, anwx[anc], anwy[anc]);
                VLOG(1)<<"Applied adaptiveNorm using window sizes "<<anwx[anc]<<" and "<<anwy[anc]<<endl;
                anc++;
                break;

            case 4:
                medianBlur(im, im2, mfkernel[mfc]);
                VLOG(1)<<"Applied medianFilter using kernel size "<<mfkernel[mfc]<<endl;
                mfc++;
                break;

            case 5:
                boxFilter(im, im2, -1, Size(sMeankernel[sMeanc], sMeankernel[sMeanc]));
                VLOG(1)<<"Applied slidingMean using kernel size "<<sMeankernel[sMeanc]<<endl;
                sMeanc++;
                break;
                
            case 6:
                slidingMinToZero(im, im2, smtzwx[smtzc], smtzwy[smtzc]);
                VLOG(1)<<"Applied slidingMinToZero using window sizes "<<smtzwx[smtzc]<<" and "<<smtzwy[smtzc]<<endl;
                smtzc++;
                break;

            }

            //qimshow(im2);
            im = im2.clone();

        }

        out = im.clone();

    } else {

        out = in.clone();

    }

}

void saRefocus::adaptiveNorm(Mat in, Mat &out, int xf, int yf) {

    // TODO: this will have to change assuming image coming in is a float

    int xs = in.cols/xf;
    int ys = in.rows/yf;

    if (xs*xf != in.cols || ys*yf != in.rows)
        LOG(WARNING)<<"Adaptive normalization divide factor leads to non integer window sizes!"<<endl;

    out.create(in.rows, in.cols, CV_8U);

    for (int i=0; i<xf; i++) {
        for (int j=0; j<yf; j++) {
            
            Mat submat = in(Rect(i*xs,j*ys,xs,ys)).clone();
            Mat subf; submat.convertTo(subf, CV_32F);
            SparseMat spsubf(subf);

            double min, max;            
            minMaxLoc(spsubf, &min, &max, NULL, NULL);
            min--;
            if (min>255.0) min = 0;
            subf -+ min; subf /= max; subf *= 255;
            subf.convertTo(submat, CV_8U);

            submat.copyTo(out(Rect(i*xs,j*ys,xs,ys)));

        }
    }

}

void saRefocus::slidingMinToZero(Mat in, Mat &out, int xf, int yf) {

    int xs = in.cols/xf;
    int ys = in.rows/yf;

    if (xs*xf != in.cols || ys*yf != in.rows)
        LOG(WARNING)<<"Sliding minimum divide factor leads to non integer window sizes!"<<endl;

    out.create(in.rows, in.cols, CV_8U);

    for (int i=0; i<xf; i++) {
        for (int j=0; j<yf; j++) {
            
            Mat submat = in(Rect(i*xs,j*ys,xs,ys)).clone();
            Mat subf; submat.convertTo(subf, CV_32F);
            SparseMat spsubf(subf);

            double min, max;            
            minMaxLoc(spsubf, &min, &max, NULL, NULL);
            min--;
            if (min>255.0) min = 0;
            subf -+ min;
            subf.convertTo(submat, CV_8U);

            submat.copyTo(out(Rect(i*xs,j*ys,xs,ys)));

        }
    }

}

void saRefocus::parse_preprocess_settings(string path) {

    ifstream file;
    file.open(path.c_str());

    string op;
    while (getline(file, op)) {

        // threshold = 1
        // gaussianBlur = 2
        // adaptiveNorm = 3
        // medianFilter = 4
        // slidingMean = 5
        // slidingMinToZero = 6

        if (op.compare("threshold")==0) {

            pp_ops.push_back(1);
            int v1;
            file>>v1;
            thresh_vals.push_back(v1);

        } else if (op.compare("gaussianBlur")==0) {

            pp_ops.push_back(2);
            int v1;   float v2;
            file>>v1; file>>v2;
            gbkernel.push_back(v1);
            gbsigma.push_back(v2);

        } else if (op.compare("adaptiveNorm")==0) {

            pp_ops.push_back(3);
            int v1;   int v2;
            file>>v1; file>>v2;
            anwx.push_back(v1);
            anwy.push_back(v2);

        } else if (op.compare("medianFilter")==0) {

            pp_ops.push_back(4);
            int v1;
            file>>v1;
            mfkernel.push_back(v1);

        } else if (op.compare("slidingMean")==0) {

            pp_ops.push_back(5);
            int v1;
            file>>v1;
            sMeankernel.push_back(v1);

        } else if (op.compare("slidingMinToZero")==0) {

            pp_ops.push_back(6);
            int v1;   int v2;
            file>>v1; file>>v2;
            smtzwx.push_back(v1);
            smtzwy.push_back(v2);

        } else {

            LOG(FATAL)<<"Invalid preprocess operation "<<op<<endl;

        }

        getline(file, op);

    } 

}

// ---Expert mode functions--- //

void saRefocus::setArrayData(vector<Mat> imgs_sub, vector<Mat> Pmats, vector<Mat> cam_locations) {

    img_size_ = Size(imgs_sub[0].cols, imgs_sub[0].rows);

    P_mats_ = Pmats;

    for (int i=0; i<imgs_sub.size(); i++) {
        
        vector<Mat> sub;
        
        // Applying a 5x5 1.5x1.5 sigma GaussianBlur to preprocess
        // Mat img;
        // GaussianBlur(imgs_sub[i], img, Size(15,15), 1.5, 1.5);

        sub.push_back(imgs_sub[i]);
        imgs.push_back(sub);
    
    }

    cam_locations_ = cam_locations;

    /*
    geom[0] = -100.0;
    geom[1] = 1.0; geom[2] = 1.5; geom[3] = 1.33;
    geom[4] = 5.0;
    */

}

void saRefocus::addView(Mat img, Mat P, Mat location) {

    img_size_ = Size(img.cols, img.rows);

    P_mats_.push_back(P);
    
    vector<Mat> sub; sub.push_back(img);
    imgs.push_back(sub);

    cam_locations_.push_back(location);

    num_cams_++;

}

void saRefocus::addViews(vector< vector<Mat> > frames, vector<Mat> Ps, vector<Mat> locations) {

    Mat img = frames[0][0];
    img_size_ = Size(img.cols, img.rows);

    P_mats_ = Ps;
    cam_locations_ = locations;

    for (int i=0; i<frames[0].size(); i++) {
        vector<Mat> view;
        for (int j=0; j<frames.size(); j++) {
            view.push_back(frames[j][i]);
        }
        imgs.push_back(view);
    }

    // imgs = frames;

    for (int i=1; i<frames.size(); i++)
        frames_.push_back(i);

    num_cams_ = frames[0].size();

}

void saRefocus::clearViews() {

    P_mats_.clear();
    imgs.clear();
    cam_locations_.clear();
    num_cams_ = 0;

}

void saRefocus::setF(double f) {

    scale_ = f;

}

void saRefocus::setMult(int flag, double exp) {

    mult_ = flag;
    mult_exp_ = exp;

}

void saRefocus::setHF(int hf) {

    CORNER_FLAG = hf;

}

void saRefocus::setRefractive(int ref, double zW, double n1, double n2, double n3, double t) {

    REF_FLAG = ref;
    geom[0] = zW;
    geom[1] = n1; geom[2] = n2; geom[3] = n3;
    geom[4] = t;

}

string saRefocus::showSettings() {

    stringstream s;
    s<<"--- FLAGS ---"<<endl;
    s<<"GPU:\t\t"<<GPU_FLAG<<endl;
    s<<"Refractive:\t"<<REF_FLAG<<endl;
    if (REF_FLAG) {
        s<<"Wall z: "<<geom[0]<<endl;
        s<<"n1: "<<geom[1]<<endl;
        s<<"n2: "<<geom[2]<<endl;
        s<<"n3: "<<geom[3]<<endl;
        s<<"Wall t: "<<geom[4]<<endl;
    }
    s<<"HF Method:\t"<<CORNER_FLAG<<endl;
    s<<"Multiplicative:\t"<<mult_<<endl;
    if (mult_)
        s<<"Mult. exp.:\t"<<mult_exp_<<endl;
    s<<endl<<"--- Other Parameters ---"<<endl;
    s<<"Num Cams:\t"<<num_cams_<<endl;
    s<<"f:\t\t"<<scale_;

    return(s.str());

}

// Python wrapper
BOOST_PYTHON_MODULE(refocusing) {

    using namespace boost::python;

    class_<saRefocus>("saRefocus")
        .def("addView", &saRefocus::addView)
        .def("clearViews", &saRefocus::clearViews)
        .def("setF", &saRefocus::setF)
        .def("setMult", &saRefocus::setMult)
        .def("setHF", &saRefocus::setHF)
        .def("setRefractive", &saRefocus::setRefractive)
        .def("showSettings", &saRefocus::showSettings)
        .def("initializeGPU", &saRefocus::initializeGPU)
        .def("refocus", &saRefocus::refocus)
    ;

}
