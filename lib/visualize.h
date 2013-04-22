#include "std_include.h"

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "tools.h"

using namespace cv;
using namespace std;

class gpuRefocus {

 public:

 gpuRefocus(vector<Mat> P_mats, vector<Mat> imgs, double scale, double z, Size img_size):
    P_mats(P_mats), array_host(imgs), scale(scale), z(z), img_size(img_size) {}

    ~gpuRefocus() {
        //delete[] array_host;
        //delete[] P_mats;
    }

    void start();
    void refocus(double z);
    void initialize();

 private:

    vector<Mat> array_host;
    vector<Mat> P_mats;
    Mat refocused_host;
    Mat den_host;

    vector<gpu::GpuMat> array;
    gpu::GpuMat temp;
    gpu::GpuMat temp2;
    gpu::GpuMat refocused;
    
    Size img_size;
    double z;
    double scale;

};

class cpuRefocus {

 public:

 cpuRefocus(vector<Mat> P_mats, vector<Mat> imgs, double scale, double z, Size img_size):
    P_mats(P_mats), array(imgs), scale(scale), z(z), img_size(img_size) {}
    
    ~cpuRefocus() {
        //delete[] array_host;
        //delete[] P_mats;
    }

    void start() {

        namedWindow("Result", CV_WINDOW_AUTOSIZE);
        refocus(z);
        
        while( 1 ){
            int key = cvWaitKey(10);
            cout<<"z = "<<z<<endl;
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

    void refocus(double z) {

        Mat H, trans;
        T_from_P(P_mats[0], H, z, scale, img_size);
        warpPerspective(array[0], trans, H, img_size);
        trans /= 255.0;
        refocused = trans.clone()/double(array.size());

        for (int i=1; i<array.size(); i++) {
            
            Mat H, trans;
            T_from_P(P_mats[i], H, z, scale, img_size);
            warpPerspective(array[i], trans, H, img_size);
            trans /= 255.0;
            refocused += trans.clone()/double(array.size());
        
        }

        imshow("Result", refocused);

    }

 private:

    vector<Mat> array;
    vector<Mat> P_mats;
    Mat refocused;
    Size img_size;
    double z, scale;

};
