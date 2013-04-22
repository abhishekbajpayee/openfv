// -------------------------------------------------------
// -------------------------------------------------------
// Synthetic Aperture - Particle Tracking Velocimetry Code
// --- Calibration IO Library ---
// -------------------------------------------------------
// Author: Abhishek Bajpayee
//         Dept. of Mechanical Engineering
//         Massachusetts Institute of Technology
// -------------------------------------------------------
// -------------------------------------------------------

#include "std_include.h"
#include "optimization.h"

using namespace cv;
using namespace std;

// Function to display a Matrix
void disp_M(Mat M) {
    
    for (int i=0; i<M.cols; i++) {
        for (int j=0; j<M.rows; j++) {
            printf("%f \t", M.at<double>(i,j));
        }
        printf("\n");
    }
    printf("\n");

}

void write_ba_result(baProblem &ba_problem, string filename) {

    ofstream file;
    file.open(filename.c_str());

    int* cameras = ba_problem.camera_index();
    int* points = ba_problem.point_index();
    int num_observations = ba_problem.num_observations();
    const double* observations = ba_problem.observations();

    int num_cameras = ba_problem.num_cameras();
    double* camera_params = ba_problem.mutable_cameras();

    int num_points = ba_problem.num_points();
    double* world_points = ba_problem.mutable_points();

    file<<num_cameras<<"\t"<<num_points<<"\t"<<num_observations<<endl;

    for (int i=0; i<ba_problem.num_observations(); i++) {
        file<<cameras[i]<<"\t";
        file<<points[i]<<"\t";
        file<<observations[2*i]<<"\t";
        file<<observations[(2*i)+1]<<endl;
    }

    for (int i=0; i<num_cameras; i++) {
        for (int j=0; j<9; j++) {
            file<<camera_params[(i*9)+j]<<endl;
        }
    }

    for (int i=0; i<num_points; i++) {
        for (int j=0; j<3; j++) {
            file<<world_points[(i*3)+j]<<endl;
        }
    }

    file<<ba_problem.cx<<endl<<ba_problem.cy<<endl<<ba_problem.scale;

    file.close();

}

// Write BA results to files so Matlab can read them and plot world points and
// camera locations
void write_ba_matlab(baProblem &ba_problem, vector<Mat> translations, vector<Mat> P_mats) {

    ofstream file, file1, file2, file3, file4, file5;
    file.open("matlab/world_points.txt");
    file1.open("matlab/camera_points.txt");
    file2.open("matlab/plane_params.txt");
    file3.open("matlab/P_mats.txt");
    file4.open("matlab/Rt.txt");
    file5.open("matlab/f.txt");

    int* cameras = ba_problem.camera_index();
    int* points = ba_problem.point_index();
    int num_observations = ba_problem.num_observations();
    const double* observations = ba_problem.observations();

    int num_cameras = ba_problem.num_cameras();
    double* camera_params = ba_problem.mutable_cameras();

    int num_points = ba_problem.num_points();
    double* world_points = ba_problem.mutable_points();

    int num_planes = ba_problem.num_planes();
    double* plane_params = ba_problem.mutable_planes();

    for (int i=0; i<num_points; i++) {
        for (int j=0; j<3; j++) {
            file<<world_points[(i*3)+j]<<"\t";
        }
        file<<endl;
    }

    for (int i=0; i<num_cameras; i++) {
        for (int j=0; j<3; j++) {
            file1<<translations[i].at<double>(0,j)<<"\t";
        }
        file1<<endl;
    }

    for (int i=0; i<num_planes; i++) {
        for (int j=0; j<4; j++) {
            file2<<plane_params[(i*4)+j]<<"\t";
        }
        file2<<endl;
    }

    for (int i=0; i<P_mats.size(); i++) {
        for (int j=0; j<3; j++) {
            for (int k=0; k<4; k++) {
                file3<<P_mats[i].at<double>(j,k)<<"\t";
            }
            file3<<endl;
        }
    }

    Mat_<double> rvec = Mat_<double>::zeros(1,3);
    Mat_<double> tvec = Mat_<double>::zeros(1,3);
    for (int i=0; i<num_cameras; i++) {
        
        file5<<camera_params[(i*9)+6]<<endl;

        for (int j=0; j<3; j++) {
            rvec.at<double>(0,j) = camera_params[(i*9)+j];
            tvec.at<double>(0,j) = camera_params[(i*9)+j+3];
        }
        
        Mat R;
        Rodrigues(rvec, R);
        
        for (int j=0; j<3; j++) {
            for (int k=0; k<3; k++) {
                file4<<R.at<double>(j,k)<<"\t";
            }
            file4<<tvec.at<double>(0,j)<<endl;
        }

    }

    file.close();
    file1.close();
    file2.close();
    file3.close();
    file4.close();
    file5.close();

}

void write_align_data(vector<Mat> rvecs, vector<Mat> tvecs, string align_file) {

    ofstream file;
    file.open(align_file.c_str());

    file<<rvecs.size()<<endl;

    for (int i=0; i<rvecs.size(); i++) {
        for (int j=0; j<3; j++) {
            file<<rvecs[i].at<double>(0,j)<<endl;
        }
        for (int j=0; j<3; j++) {
            file<<tvecs[i].at<double>(0,j)<<endl;
        }
    }

    file.close();

}

void write_aligned(alignProblem align_problem, string aligned_file) {

    ofstream file, file1;
    file.open("matlab/aligned_rt.txt");
    file1.open(aligned_file.c_str());


    double* params = align_problem.mutable_params();
    int num_cams = align_problem.num_cameras();

    Mat_<double> rvec = Mat_<double>::zeros(1,3);
    Mat_<double> tvec = Mat_<double>::zeros(1,3);
    for (int i=0; i<num_cams; i++) {
        cout<<i<<endl;
        for (int j=0; j<3; j++) {
            rvec.at<double>(0,j) = params[(i*6)+j];
            tvec.at<double>(0,j) = params[(i*6)+j+3];
        }
        
        Mat R;
        Rodrigues(rvec, R);
        
        for (int j=0; j<3; j++) {
            for (int k=0; k<3; k++) {
                file<<R.at<double>(j,k)<<"\t";
            }
            file<<tvec.at<double>(0,j)<<endl;
        }
        
    }

    for (int i=0; i<6*num_cams; i++) {
        file1<<params[i]<<endl;
    }
        
    file.close();
    file1.close();

}

void write_calib_result(baProblem ba_problem, vector<string> cam_names, Size img_size, double pix_per_phys, string path) {
    /*
    ofstream file;
    string filename = path+"calibration_result.txt";
    file.open(filename.c_str());
    
    file<<cam_names.size()<<endl;

    for (int i=0; i<cam_names.size(); i++) {
        file<<cam_names[i]<<endl;
    }
    /*
    int num_cameras = ba_problem.num_cameras();
    double* camera_params = ba_problem.mutable_cameras();

    file<<num_cameras<<endl;
    for (int i=0; i<num_cameras; i++) {
        for (int j=0; j<9; j++) {
            file<<camera_params[(i*9)+j]<<endl;
        }
    }

    file<<(img_size.width*0.5)<<endl<<(img_size.height*0.5)<<endl<<pix_per_phys;
    
    file.close();*/

}

void write_align_data(baProblem &ba_problem, string filename, vector< vector<Mat> > translations_new, char* argv) {

    ofstream file;
    file.open(filename.c_str());

    int* cameras = ba_problem.camera_index();
    int* points = ba_problem.point_index();
    int num_observations = ba_problem.num_observations();
    const double* observations = ba_problem.observations();

    int num_cameras = ba_problem.num_cameras();
    double* camera_params = ba_problem.mutable_cameras();

    int num_points = ba_problem.num_points();
    double* world_points = ba_problem.mutable_points();

    double ox1, oy1, oz1;
    ox1 = 0;
    oy1 = 0;
    oz1 = 0;
    for (int i=0; i<num_points; i++) {
        ox1 += world_points[i*3];
        oy1 += world_points[(i*3)+1];
        oz1 += world_points[(i*3)+2];
    }
    ox1 = ox1/double(num_points);
    oy1 = oy1/double(num_points);
    oz1 = oz1/double(num_points);

    cout<<ox1<<"\t"<<oy1<<"\t"<<oz1<<endl;

    double ox2, oy2, oz2;
    ox2 = 0;
    oy2 = 0;
    oz2 = 0;
    for (int i=0; i<num_cameras; i++) {
        ox2 += translations_new[i][0].at<double>(0,0);
        oy2 += translations_new[i][0].at<double>(0,1);
        oz2 += translations_new[i][0].at<double>(0,2);
    }
    ox2 = (ox2/double(num_cameras)) - ox1;
    oy2 = (oy2/double(num_cameras)) - oy1;
    oz2 = (oz2/double(num_cameras)) - oz1;
    cout<<ox2<<"\t"<<oy2<<"\t"<<oz2<<endl;
    double den = sqrt(pow(ox2, 2)+pow(oy2, 2)+pow(oz2, 2));
    ox2 /= den;
    oy2 /= den;
    oz2 /= den;

    vector<Mat> translations;
    for (int i=0; i<translations_new.size(); i++) {
        translations.push_back(translations_new[i][0]);
    }

    ofstream plane_file;
    string plane_filename("temp/plane_fitting.txt");
    plane_file.open(plane_filename.c_str());
    plane_file<<4<<endl;
    
    // Initial values for plane estimation
    plane_file<<ox2<<endl;
    plane_file<<oy2<<endl;
    plane_file<<oz2<<endl;
    plane_file<<-(ox2*translations[0].at<double>(0,0) + oy2*translations[0].at<double>(0,1) + oz2*translations[0].at<double>(0,2))<<endl;
    plane_file.close();

    leastSquares ls_problem;
    cout<<"FINAL ERROR: "<<fit_plane(ls_problem, plane_filename, argv, translations);
    
    cout<<endl;
    double* params = ls_problem.mutable_params();
    for (int i=0; i<4; i++) {
        cout<<params[0]<<"\t";
    }
    cout<<endl;

}
