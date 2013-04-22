// -------------------------------------------------------
// -------------------------------------------------------
// Synthetic Aperture - Particle Tracking Velocimetry Code
// --- Calibration Processing Library ---
// -------------------------------------------------------
// Author: Abhishek Bajpayee
//         Dept. of Mechanical Engineering
//         Massachusetts Institute of Technology
// -------------------------------------------------------
// -------------------------------------------------------

#include "std_include.h"

using namespace cv;
using namespace std;

// Function to calculate T (translation matrix) from R and t matrices for all cameras AND images
void get_translations_all(vector< vector<Mat> > rvecs, vector< vector<Mat> > tvecs, vector< vector<Mat> > &translations) {

    for (int i=0; i<rvecs.size(); i++) {
        
        vector<Mat> translations_cam;
        Mat R, R_t;

        for (int j=0; j<rvecs[0].size(); j++) {
            
            Mat T;
            Rodrigues(rvecs[i][j], R);
            transpose(R, R_t);
            T = -R_t*tvecs[i][j];
            translations_cam.push_back(T.clone());

        }

        translations.push_back(translations_cam);

    }

}

// Function to calculate T (translation matrix) from R and t matrices for all cameras
void get_translations(vector<Mat> rvecs, vector<Mat> tvecs, vector<Mat> &translations) {

    for (int i=0; i<rvecs.size(); i++) {
        
        Mat translation_cam;
        Mat R, R_t, T;
       
        Rodrigues(rvecs[i], R);
        transpose(R, R_t);
        T = -R_t*tvecs[i];

        translations.push_back(T.clone());

    }

}

// Construct a P matrix from K, R and T matrices
Mat P_from_KRT(Mat K, Mat rvec, Mat tvec, Mat rmean) {

    Mat R_o;
    Rodrigues(rvec, R_o);

    Mat rmean_t;
    transpose(rmean, rmean_t);

    Mat R = R_o*rmean_t;

    Mat_<double> P = Mat_<double>::zeros(3,4);
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            P(i,j) = R.at<double>(i,j);
        }
        P(i,3) = tvec.at<double>(0,i);
    }

    P = K*P;
                                              
    return(P);

}
