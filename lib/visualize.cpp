#include "std_include"

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "tools.h"

using namespace cv;
using namespace std;

void gpuRefocus::start() {

    initialize();

    namedWindow("Result", CV_WINDOW_AUTOSIZE);       
    refocus(z);
    
    while( 1 ){
        int key = cvWaitKey(10);
        if( (key & 255)==83 ) {
            z += 0.5;
            refocus(z);
        } else if( (key & 255)==81 ) {
            z -= 0.5;
            refocus(z);
        } else if( (key & 255)==27 ) {
            break;
        }
    }

}

void gpuRefocus::refocus(double z) {

    Scalar fact = Scalar(1/double(array.size()));

    Mat H, trans;
    T_from_P(P_mats[0], H, z, scale, img_size);
    gpu::warpPerspective(array[0], temp, H, array[0].size());
    gpu::multiply(temp, fact, temp2);
    
    refocused = temp2.clone();
    
    for (int i=1; i<array_host.size(); i++) {
        
        T_from_P(P_mats[i], H, z, scale, img_size);
        
        gpu::warpPerspective(array[i], temp, H, img_size);
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

void gpuRefocus::initialize() {

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
