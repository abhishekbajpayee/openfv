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

    cout<<"\nREADING IMAGES TO REFOCUS...\n\n";

    for (int i=0; i<num_cams_; i++) {

        cout<<"Camera "<<i+1<<" of "<<num_cams_<<"...";

        string path_tmp;
        vector<Mat> refocusing_imgs_sub;

        path_tmp = path+cam_names_[i]+"/"+img_prefix;

        dir = opendir(path_tmp.c_str());

        while(ent = readdir(dir)) {
            temp_name = ent->d_name;
            if (temp_name.compare(dir1)) {
                if (temp_name.compare(dir2)) {
                    string path_img = path_tmp+temp_name;
                    image = imread(path_img, 0);
                    image.convertTo(fimage, CV_32F);
                    refocusing_imgs_sub.push_back(fimage.clone());
                }
            }
        }

        imgs.push_back(refocusing_imgs_sub);
        path_tmp = "";

        cout<<"done!\n";
   
    }
 
    cout<<"\nDONE READING IMAGES!\n\n";

}

void saRefocus::GPUliveView() {

    initializeGPU();

    namedWindow("Result", CV_WINDOW_AUTOSIZE);       
    GPUrefocus(z, thresh, 1);
    
    double dz = 0.5;
    double dthresh = 5;

    while( 1 ){
        int key = cvWaitKey(10);
        if( (key & 255)==83 ) {
            z += dz;
            GPUrefocus(z, thresh, 1);
        } else if( (key & 255)==81 ) {
            z -= dz;
            GPUrefocus(z, thresh, 1);
        } else if( (key & 255)==82 ) {
            if (thresh<255) { 
                thresh += dthresh; 
                GPUrefocus(z, thresh, 1); 
            }
        } else if( (key & 255)==84 ) {
            if (thresh>0) { 
                thresh -= dthresh; 
                GPUrefocus(z, thresh, 1); 
            }
        } else if( (key & 255)==27 ) {
            break;
        }
    }

}

// TODO: Right now this function only uploads first time step
//       so change to upload more time steps so this can be controlled
//       from outside
void saRefocus::initializeGPU() {

    cout<<endl<<"INITIALIZING GPU FOR VISUALIZATION..."<<endl;
    cout<<"CUDA Enabled GPU Devices: "<<gpu::getCudaEnabledDeviceCount<<endl;
    
    gpu::DeviceInfo gpuDevice(gpu::getDevice());
    
    cout<<"---"<<gpuDevice.name()<<"---"<<endl;
    cout<<"Total Memory: "<<(gpuDevice.totalMemory()/pow(1024.0,2))<<" MB"<<endl;
    cout<<"Free Memory: "<<(gpuDevice.freeMemory()/pow(1024.0,2))<<" MB"<<endl;
    
    int t=0;
    cout<<"Uploading images to GPU..."<<endl;
    for (int i=0; i<num_cams_; i++) {
        temp.upload(imgs[i][t]);
        array.push_back(temp.clone());
    }
    cout<<"Free Memory: "<<(gpuDevice.freeMemory()/pow(1024.0,2))<<" MB"<<endl;

}


void saRefocus::GPUrefocus(double z, double thresh, int live) {

    Scalar fact = Scalar(1/double(array.size()));

    Mat H, trans;
    T_from_P(P_mats_[0], H, z, scale_, img_size_);
    gpu::warpPerspective(array[0], temp, H, array[0].size());
    gpu::multiply(temp, fact, temp2);
    
    refocused = temp2.clone();
    
    for (int i=1; i<num_cams_; i++) {
        
        T_from_P(P_mats_[i], H, z, scale_, img_size_);
        
        gpu::warpPerspective(array[i], temp, H, img_size_);
        gpu::multiply(temp, fact, temp2);
        gpu::add(refocused, temp2, refocused);        
        
    }
    
    gpu::threshold(refocused, refocused, thresh, 0, THRESH_TOZERO);

    Mat refocused_host_(refocused);
    refocused_host_ /= 255.0;

    if (live) {
        char title[50];
        sprintf(title, "z = %f, thresh = %f", z, thresh);
        putText(refocused_host_, title, Point(10,20), FONT_HERSHEY_PLAIN, 1.0, Scalar(255,0,0));
        imshow("Result", refocused_host_);
    }

}
