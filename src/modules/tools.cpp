#ifndef TOOLS_LIBRARY
#define TOOLS_LIBRARY

#include "std_include.h"
#include "tools.h"

using namespace std;
using namespace cv;

void init(int argc, char** argv) {

    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    google::InstallFailureFunction(&failureFunction);
    FLAGS_logtostderr=1;

}

void T_from_P(Mat P, Mat &H, double z, double scale, Size img_size) {

    Mat_<double> A = Mat_<double>::zeros(3,3);

    for (int i=0; i<3; i++) {
        for (int j=0; j<2; j++) {
            A(i,j) = P.at<double>(i,j);
        }
    }

    for (int i=0; i<3; i++) {
        A(i,2) = P.at<double>(i,2)*z+P.at<double>(i,3);
    }

    Mat A_inv = A.inv();

    Mat_<double> D = Mat_<double>::zeros(3,3);
    D(0,0) = scale;
    D(1,1) = scale;
    D(2,2) = 1;
    D(0,2) = img_size.width*0.5;
    D(1,2) = img_size.height*0.5;
    
    Mat T = D*A_inv;
    
    H = T.clone();

}

bool dirExists(string dirPath) {

    if ( dirPath.c_str() == NULL) return false;

    DIR *pDir;
    bool bExists = false;

    pDir = opendir (dirPath.c_str());

    if (pDir != NULL)
    {
        bExists = true;    
        (void) closedir (pDir);
    }

    return bExists;

}

// Function to calculate mean of any matrix
// Returns 1 if success
int matrixMean(vector<Mat> mats_in, Mat &mat_out) {

    if (mats_in.size()==0) {
        cout<<"\nInput matrix vector empty!\n";
        return 0;
    }

    for (int i=0; i<mats_in.size(); i++) {
        for (int j=0; j<mats_in[0].rows; j++) {
            for (int k=0; k<mats_in[0].cols; k++) {
                mat_out.at<double>(j,k) += mats_in[i].at<double>(j,k);
            }
        }
    }

    mat_out = mat_out/double(mats_in.size());
    
    return 1;

}

// Construct aligned and unaligned P matrix from K, R and T matrices
Mat P_from_KRT(Mat K, Mat rvec, Mat tvec, Mat rmean, Mat &P_u, Mat &P) {

    Mat rmean_t;
    transpose(rmean, rmean_t);

    Mat R = rvec*rmean_t;
    
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            P_u.at<double>(i,j) = rvec.at<double>(i,j);
            P.at<double>(i,j) = R.at<double>(i,j);
        }
        P_u.at<double>(i,3) = tvec.at<double>(0,i);
        P.at<double>(i,3) = tvec.at<double>(0,i);
    }
    
    P_u = K*P_u;
    P = K*P;

}

double dist(Point3f p1, Point3f p2) {

    double distance = sqrt(pow(p2.x-p1.x,2) + pow(p2.y-p1.y,2) + pow(p2.z-p1.z,2));
    
    return(distance);

}

void qimshow(Mat image) {

    namedWindow("Image", CV_WINDOW_AUTOSIZE);
    imshow("Image", image);
    waitKey(0);
    destroyWindow("Image");

}

void pimshow(Mat image, double z, int n) {

    namedWindow("Image", CV_WINDOW_AUTOSIZE);
    
    char title[50];
    sprintf(title, "z = %f, n = %d", z, n);
    putText(image, title, Point(10,20), FONT_HERSHEY_PLAIN, 1.0, Scalar(255,0,0));
    imshow("Image", image);
    
    waitKey(0);
    destroyWindow("Image");

}

Mat getRotMat(double x, double y, double z) {

    double pi = 3.14159;
    x = x*pi/180.0;
    y = y*pi/180.0;
    z = z*pi/180.0;

    Mat_<double> Rx = Mat_<double>::zeros(3,3);
    Mat_<double> Ry = Mat_<double>::zeros(3,3);
    Mat_<double> Rz = Mat_<double>::zeros(3,3);

    Rx(0,0) = 1;
    Rx(1,1) = cos(x);
    Rx(1,2) = -sin(x);
    Rx(2,1) = sin(x);
    Rx(2,2) = cos(x);

    Ry(0,0) = cos(y);
    Ry(1,1) = 1;
    Ry(2,0) = -sin(y);
    Ry(2,2) = cos(y);
    Ry(0,2) = sin(y);

    Rz(0,0) = cos(z);
    Rz(0,1) = -sin(z);
    Rz(1,0) = sin(z);
    Rz(1,1) = cos(z);
    Rz(2,2) = 1;

    Mat R = Rx*Ry*Rz;

    return(R);

}

void failureFunction() {

    LOG(INFO)<<"Good luck debugging that X-|";
    exit(1);

}

#endif
