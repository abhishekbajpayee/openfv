#include "std_include.h"

using namespace cv;
using namespace std;

// Function to refocus an image at depth z
// Work: - write eventually so that P_mats are read from previously saved
//         calibration results; function right now assumes cam_names are same
//         as camera ids
void refocus_img(vector< vector<Mat> > refocusing_imgs, vector<Mat> P_mats, double pix_per_phys, double z, int method) {

    Size img_size = Size(refocusing_imgs[0][0].cols, refocusing_imgs[0][0].rows);
    Mat H, transformed, refocused, image, power;
    int num_cams = refocusing_imgs.size();

    if (method==0) {
        refocused = refocusing_imgs[0][0].clone();
        refocused = Scalar(0);
        for (int i=0; i<num_cams; i++) {
            Mat_<double> H = Mat_<double>::zeros(3,3);
            Mat transformed;
            T_from_P(P_mats[i], H, z, pix_per_phys, img_size);
            warpPerspective(refocusing_imgs[i][0], transformed, H, img_size);
            transformed /= 255.0;
            refocused += transformed.clone()/double(num_cams);
        }
    } else if (method==1) {
        
        double exp = 1/refocusing_imgs[0].size();
        Mat_<double> H = Mat_<double>::zeros(3,3);
        Mat transformed;
        T_from_P(P_mats[0], H, z, pix_per_phys, img_size);
        warpPerspective(refocusing_imgs[0][0], transformed, H, img_size);
        transformed /= 255.0;
        pow(transformed, exp, refocused);
        imshow("img", refocused);
        waitKey(0);
        
        for (int i=1; i<num_cams; i++) {
            Mat_<double> H = Mat_<double>::zeros(3,3);
            Mat transformed;
            T_from_P(P_mats[i], H, z, pix_per_phys, img_size);
            warpPerspective(refocusing_imgs[i][0], transformed, H, img_size);
            transformed /= 255.0;
            pow(transformed, exp, power);
            imshow("img", power);
            waitKey(0);
            refocused = refocused.mul(power);
        }

    }

    imshow("Image", refocused);
    waitKey(0);

}

// Function to calculate P matrices from calibration data
// Work - account for distortion if any
void get_P_mats(vector<Mat> cameraMats, vector< vector<Mat> > rvecs, vector< vector<Mat> > tvecs, vector<Mat> &P_mats) {

    for (int i=0; i<cameraMats.size(); i++) {

        Mat_<double> Rt = Mat_<double>::zeros(3,4);
        
        Mat R;
        Rodrigues(rvecs[i][0], R);

        for (int j=0; j<3; j++) {
            for (int k=0; k<3; k++) {
                Rt(j,k) = R.at<double>(j,k);
            }
            Rt(j,3) = tvecs[i][0].at<double>(j,0);
        }

        Mat P = cameraMats[i]*Rt;

        P_mats.push_back(P.clone());

    }

}

// Function to calculate H matrices
void get_H_mats(vector< vector<Mat> > rvecs, vector< vector<Mat> > tvecs, vector< vector<Mat> > translations, double scale, vector<Mat> &H_mats) {

    Mat_<double> n = Mat_<double>::zeros(1,3);
    n(0,2) = 1;

    Mat_<double> r_ref = Mat_<double>::zeros(3,1);
    Mat_<double> t_ref = Mat_<double>::zeros(3,1);
    t_ref(2,0) = 200;

    for (int i=0; i<rvecs.size(); i++) {

        Mat R;
        Rodrigues(rvecs[i][0]-r_ref, R);

        Mat A;
        A = (tvecs[i][0]-t_ref)*n;
        double d = translations[i][0].at<double>(0,2);
        A = (1/abs(d))*A;

        Mat H = R - A;

        H_mats.push_back(H.clone());
        cout<<H<<endl;

    }

}
