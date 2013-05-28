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

using namespace std;
using namespace cv;

void saRefocus::read_imgs(string path) {

    DIR *dir;
    struct dirent *ent;
 
    string dir1(".");
    string dir2("..");
    string temp_name;
    string img_prefix = "";

    Mat image, fimage;

    vector<string> img_names;

    cout<<"\nREADING IMAGES TO REFOCUS...\n\n";

    for (int i=0; i<num_cams_; i++) {

        cout<<"Camera "<<i+1<<" of "<<num_cams_<<"..."<<endl;

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
        for (int i=0; i<img_names.size(); i++) {
            //cout<<i<<": "<<img_names[i]<<endl;
            image = imread(img_names[i], 0);
            image.convertTo(fimage, CV_32F);
            refocusing_imgs_sub.push_back(fimage.clone());
        }
        img_names.clear();

        imgs.push_back(refocusing_imgs_sub);
        path_tmp = "";

        cout<<"done!\n";
   
    }
 
    cout<<"\nDONE READING IMAGES!\n\n";

}

void saRefocus::GPUliveView() {

    initializeGPU();

    active_frame_ = 0;

    namedWindow("Result", CV_WINDOW_AUTOSIZE);       
    GPUrefocus(z, thresh, 1, active_frame_);
    
    double dz = 0.5;
    double dthresh = 5;

    while( 1 ){
        int key = cvWaitKey(10);
        //cout<<(key & 255)<<endl;
        if( (key & 255)==83 ) {
            z += dz;
            GPUrefocus(z, thresh, 1, active_frame_);
        } else if( (key & 255)==81 ) {
            z -= dz;
            GPUrefocus(z, thresh, 1, active_frame_);
        } else if( (key & 255)==82 ) {
            if (thresh<255) { 
                thresh += dthresh; 
                GPUrefocus(z, thresh, 1, active_frame_); 
            }
        } else if( (key & 255)==84 ) {
            if (thresh>0) { 
                thresh -= dthresh; 
                GPUrefocus(z, thresh, 1, active_frame_); 
            }
        } else if( (key & 255)==46 ) {
            if (active_frame_<array_all.size()) { 
                active_frame_++; 
                GPUrefocus(z, thresh, 1, active_frame_); 
            }
        } else if( (key & 255)==44 ) {
            if (active_frame_<array_all.size()) { 
                active_frame_--; 
                GPUrefocus(z, thresh, 1, active_frame_); 
            }
        } else if( (key & 255)==27 ) {
            break;
        }
    }

}

// TODO: This function prints free memory on GPU and then
//       calls uploadToGPU() which uploads either a given
//       frame or all frames to GPU depending on frame_
void saRefocus::initializeGPU() {

    cout<<endl<<"INITIALIZING GPU FOR VISUALIZATION..."<<endl;
    cout<<"CUDA Enabled GPU Devices: "<<gpu::getCudaEnabledDeviceCount<<endl;
    
    gpu::DeviceInfo gpuDevice(gpu::getDevice());
    
    cout<<"---"<<gpuDevice.name()<<"---"<<endl;
    cout<<"Total Memory: "<<(gpuDevice.totalMemory()/pow(1024.0,2))<<" MB"<<endl;
    cout<<"Free Memory: "<<(gpuDevice.freeMemory()/pow(1024.0,2))<<" MB"<<endl;

    uploadToGPU();

}

// TODO: Right now this function just starts uploading images
//       without checking if there is enough free memory on GPU
//       or not.
void saRefocus::uploadToGPU() {

    gpu::DeviceInfo gpuDevice(gpu::getDevice());
    double free_mem_GPU = gpuDevice.freeMemory()/pow(1024.0,2);
    cout<<"Free Memory before: "<<free_mem_GPU<<" MB"<<endl;

    double factor = 0.9;

    if (frame_>=0) {

        cout<<"Uploading "<<(frame_+1)<<"th frame to GPU..."<<endl;
        for (int i=0; i<num_cams_; i++) {
            temp.upload(imgs[i][frame_]);
            array.push_back(temp.clone());
        }
        array_all.push_back(array);

    } else if (frame_==-1) {
        
        cout<<"Uploading all frame to GPU..."<<endl;
        for (int i=0; i<imgs[0].size(); i++) {
            for (int j=0; j<num_cams_; j++) {
                temp.upload(imgs[j][i]);
                array.push_back(temp.clone());
            }
            array_all.push_back(array);
            array.clear();
        }
        
    } else {
        cout<<"Invalid frame value to visualize!"<<endl;
    }

    cout<<"Free Memory after: "<<(gpuDevice.freeMemory()/pow(1024.0,2))<<" MB"<<endl;

}


void saRefocus::GPUrefocus(double z, double thresh, int live, int frame) {

    z *= warp_factor_;

    Scalar fact = Scalar(1/double(array_all[frame].size()));

    Mat H, trans;
    T_from_P(P_mats_[0], H, z, scale_, img_size_);
    gpu::warpPerspective(array_all[frame][0], temp, H, img_size_);

    if (mult_) {
        gpu::pow(temp, mult_exp_, temp2);
    } else {
        gpu::multiply(temp, fact, temp2);
    }

    refocused = temp2.clone();

    for (int i=1; i<num_cams_; i++) {
        
        T_from_P(P_mats_[i], H, z, scale_, img_size_);
        
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

    Mat refocused_host_(refocused);
    //refocused_host_ /= 255.0;

    if (live) {
        refocused_host_ /= 255.0;
        char title[50];
        sprintf(title, "z = %f, thresh = %f, frame = %d", z/warp_factor_, thresh, frame);
        putText(refocused_host_, title, Point(10,20), FONT_HERSHEY_PLAIN, 1.0, Scalar(255,0,0));
        line(refocused_host_, Point(646,482-5), Point(646,482+5), Scalar(255,0,0));
        line(refocused_host_, Point(646-5,482), Point(646+5,482), Scalar(255,0,0));
        imshow("Result", refocused_host_);
    }

    refocused_host_.convertTo(result, CV_8U);

}
