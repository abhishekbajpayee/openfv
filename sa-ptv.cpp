#include "std_include.h"
//#include "lib_include.h"

#include "calibration.h"
#include "refocusing.h"
#include "tools.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    
    // Camera Calibration Section
    clock_t begin = clock();

    string path("../experiment/calibration_rect/"); // Folder where calibration images lie
    Size grid_size = Size(6,5); // Format (horizontal_corners, vertical_corners)
    double grid_size_phys = 5;  // in [mm]

    multiCamCalibration calibration(path, grid_size, grid_size_phys);
    calibration.run();

    saRefocus refocus(calibration.refocusing_params());

    /*
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
    
    gpuRefocus session(calibration.P_mats_, imgs, calibration.pix_per_phys_, 0, calibration.img_size_);
    session.start();
    }
    */

    clock_t end = clock();
    cout<<endl<<"TIME TAKEN: "<<(float(end-begin)/CLOCKS_PER_SEC)<<" seconds"<<endl;
    return 1;

}
