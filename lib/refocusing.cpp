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

void saRefocus::startGPUsession() {

    initializeGPU();

    namedWindow("Result", CV_WINDOW_AUTOSIZE);       
    GPUrefocus(z);
    
    while( 1 ){
        int key = cvWaitKey(10);
        if( (key & 255)==83 ) {
            z += 0.5;
            GPUrefocus(z);
        } else if( (key & 255)==81 ) {
            z -= 0.5;
            GPUrefocus(z);
        } else if( (key & 255)==27 ) {
            break;
        }
    }

}

void saRefocus::initializeGPU() {

    cout<<endl<<"INITIALIZING GPU FOR VISUALIZATION..."<<endl;
    cout<<"CUDA Enabled GPU Devices: "<<gpu::getCudaEnabledDeviceCount<<endl;
    
    gpu::DeviceInfo gpuDevice(gpu::getDevice());
    
    cout<<"---"<<gpuDevice.name()<<"---"<<endl;
    cout<<"Total Memory: "<<(gpuDevice.totalMemory()/pow(1024.0,2))<<" MB"<<endl;
    cout<<"Free Memory: "<<(gpuDevice.freeMemory()/pow(1024.0,2))<<" MB"<<endl;
    cout<<"Cores: "<<gpuDevice.multiProcessorCount()<<endl;
    
    for (int i=0; i<array_host.size(); i++) {
        temp.upload(array_host[i]);
        array.push_back(temp.clone());
    }

}


void saRefocus::GPUrefocus(double z) {

    Scalar fact = Scalar(1/double(array.size()));

    Mat H, trans;
    T_from_P(P_mats_[0], H, z, scale_, img_size_);
    gpu::warpPerspective(array[0], temp, H, array[0].size());
    gpu::multiply(temp, fact, temp2);
    
    refocused = temp2.clone();
    
    for (int i=1; i<array_host.size(); i++) {
        
        T_from_P(P_mats_[i], H, z, scale_, img_size_);
        
        gpu::warpPerspective(array[i], temp, H, img_size_);
        gpu::multiply(temp, fact, temp2);
        gpu::add(refocused, temp2, refocused);        
        
    }
    
    Mat refocused_host(refocused);
    refocused_host /= 255.0;
    char title[20];
    sprintf(title, "z = %f", z);
    putText(refocused_host, title, Point(10,20), FONT_HERSHEY_PLAIN, 1.0, Scalar(255,0,0));
    imshow("Result", refocused_host);

}
