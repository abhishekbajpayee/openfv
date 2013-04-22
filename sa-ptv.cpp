#include "std_include.h"
//#include "lib_include.h"

#include "visualize.h"
#include "calibration.h"
#include "tools.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    
    // Camera Calibration Section
    clock_t begin = clock();

    string path("../experiment/calibration_skew/"); // Folder where calibration images lie
    Size grid_size = Size(6,5); // Format (horizontal_corners, vertical_corners)
    double grid_size_phys = 5;  // in [mm]

    multiCamCalibration calibration(path, grid_size, grid_size_phys);
    calibration.run();
    
    /*
    write_ba_result(ba_problem, ba_result_file);
    
    //write_calib_result(ba_problem, cam_names, img_size, pix_per_phys, path);
    
    // Reading calibration 1st round data
    vector<Mat> P_mats, rvecs2, tvecs2, translations;
    double scale;
    read_calib_data(ba_result_file, P_mats, rvecs2, tvecs2, translations, scale);
    
    //write_ba_matlab(ba_problem, translations, P_mats);
    
    int refocus = 1;
    if (refocus) {
    vector< vector<Mat> > refoc_imgs;
    string refoc_path("../experiment/objects/");
    read_refocusing_imgs(refoc_path, cam_names, refoc_imgs);

    //refocus_img(refoc_imgs, P_mats, scale, 0, 0);
    
    vector<Mat> imgs;
    for (int i=0; i<refoc_imgs.size(); i++) {
        imgs.push_back(refoc_imgs[i][0].clone());
    }
    
    gpuRefocus session(P_mats, imgs, scale, 0, img_size);
    session.start();
    }
    */

    clock_t end = clock();
    cout<<endl<<"TIME TAKEN: "<<(float(end-begin)/CLOCKS_PER_SEC)<<" seconds"<<endl;
    return 1;

}
