// -------------------------------------------------------
// -------------------------------------------------------
// Synthetic Aperture - Particle Tracking Velocimetry Code
// --- Refocusing IO Library ---
// -------------------------------------------------------
// Author: Abhishek Bajpayee
//         Dept. of Mechanical Engineering
//         Massachusetts Institute of Technology
// -------------------------------------------------------
// -------------------------------------------------------

#include "std_include.h"
//#include "optimization.h"

#include "processing_calibrate.h"

using namespace cv;
using namespace std;

// Function to read calibration result data from file written after bundle adjustment
// and then to calculate P matrices and translation matrices
void read_calib_data(string filename, vector<Mat> &P_mats, vector<Mat> &rvecs, vector<Mat> &tvecs, vector<Mat> &translations, double &scale) {
    
    int num_cameras, num_points, num_observations;
    double cx, cy, dump, tmp;
    vector<double> params;

    // Reading file for data
    ifstream file;
    file.open(filename.c_str());
    file>>num_cameras>>num_points>>num_observations;
    for (int i=0; i<num_observations; i++) {
        file>>dump>>dump>>dump>>dump;
    }
    for (int i=0; i<num_cameras; i++) {
        for (int j=0; j<9; j++) {
            file>>tmp;
            params.push_back(tmp);
        }
    }
    for (int i=0; i<num_points; i++) {
        for (int j=0; j<3; j++) {
            file>>dump;
        }
    }
    file>>cx>>cy>>scale;
    file.close();

    vector<double> params_align;
    file.open("ba_files/scene_aligned_r.txt");
    for (int i=0; i<6*num_cameras; i++) {
        file>>tmp;
        params_align.push_back(tmp);
    }
    
    //vector<Mat> rvecs;
    //vector<Mat> tvecs;
    vector<Mat> arvecs;
    vector<Mat> atvecs;

    Mat_<double> rmean = Mat_<double>::zeros(3,3);
    Mat R;
    
    vector<Mat> rvecs;
    vector<Mat> tvecs;
    
    for (int i=0; i<num_cameras; i++) {
        
        Mat_<double> rvec = Mat_<double>::zeros(1,3);
        Mat_<double> tvec = Mat_<double>::zeros(3,1);

        Mat_<double> arvec = Mat_<double>::zeros(1,3);
        Mat_<double> atvec = Mat_<double>::zeros(3,1);
        
        for (int j=0; j<3; j++) { 
            rvec(0,j) = params[(i*9)+j];  
            tvec(j,0) = params[(i*9)+j+3];

            arvec(0,j) = params_align[(i*6)+j];
            atvec(j,0) = params_align[(i*6)+j+3];
        }

        Rodrigues(rvec, R);
        rmean += R.clone()/9;

        rvecs.push_back(rvec);
        tvecs.push_back(tvec);

        arvecs.push_back(rvec);
        atvecs.push_back(tvec);

    }
    
    get_translations(rvecs, tvecs, translations);

    for (int i=0; i<num_cameras; i++) {

        Mat_<double> K = Mat_<double>::zeros(3,3);

        }

        rvecs.push_back(rvec);
        tvecs.push_back(tvec);

    }
    
    get_translations(rvecs, tvecs, translations);
    
    vector<Mat> R;
    for (int i=0; i<rvecs.size(); i++) {

        Mat Rot;
        Rodrigues(rvecs[i], Rot);
        R.push_back(Rot.clone());

    }
    
    for (int i=0; i<num_cameras; i++) {

        Mat_<double> P_mat = Mat_<double>::zeros(3,4);
        Mat_<double> K = Mat_<double>::zeros(3,3);

        for (int j=0; j<3; j++) {
            for (int k=0; k<3; k++) {
                P_mat(j,k) = R[i].at<double>(j,k);
            }
            P_mat(j,3) = tvecs[i].at<double>(0,j);
        }
        
        K(0,0) = params[(i*9)+6];
        K(1,1) = K(0,0);
        K(2,2) = 1.0;
        K(0,2) = cx;
        K(1,2) = cy;
        
        Mat P_mat = P_from_KRT(K, arvecs[i], atvecs[i], rmean);
        
        P_mats.push_back(P_mat.clone());
        
        P_mats.push_back(P_mat);
        
    }
    
}

// Function to read images to refocus
// Work: - Generalize eventually to use camera names and path
//       - Maybe include an option to limit how many to read
//         for large data sets
//       - 
void read_refocusing_imgs(string path, vector<string> cam_names, vector< vector<Mat> > &refocusing_imgs) {

    DIR *dir;
    struct dirent *ent;
 
    string dir1(".");
    string dir2("..");
    string temp_name;
    string img_prefix = "";

    Mat image, fimage;

    cout<<"\nREADING IMAGES TO REFOCUS...\n\n";

    for (int i=0; i<cam_names.size(); i++) {

        cout<<"Camera "<<i+1<<" of "<<cam_names.size()<<"...";

        string path_tmp;
        vector<Mat> refocusing_imgs_sub;

        path_tmp = path+cam_names[i]+"/"+img_prefix;

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

        refocusing_imgs.push_back(refocusing_imgs_sub);
        path_tmp = "";

        cout<<"done!\n";
   
    }
 
    cout<<"\nDONE READING IMAGES!\n\n";

}
