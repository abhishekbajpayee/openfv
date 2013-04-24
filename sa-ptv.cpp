#include "std_include.h"
//#include "lib_include.h"

#include "calibration.h"
#include "refocusing.h"
#include "pLoc.h"
#include "tools.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    
    // Camera Calibration Section
    clock_t begin = clock();

    string calib_path("../experiment/calibration_rect/"); // Folder where calibration images lie
    Size grid_size = Size(6,5); // Format (horizontal_corners, vertical_corners)
    double grid_size_phys = 5;  // in [mm]

    multiCamCalibration calibration(calib_path, grid_size, grid_size_phys);
    calibration.run();

    string refoc_path("../experiment/piv_sim_500/");

    saRefocus refocus(calibration.refocusing_params());
    refocus.read_imgs(refoc_path);
    //refocus.GPUliveView();
    refocus.initializeGPU();
    refocus.GPUrefocus(0, 50.0, 0);

    vector<Point2f> points;
    vector<Point2f> points_spix;
    Mat result;
    Mat image = refocus.refocused_host();

    pLocalize localizer(2);
    localizer.find_particles(image, points);
    localizer.refine_subpixel(image, points, points_spix);
    localizer.draw_points(image, result, points_spix);
    imshow("result", result);

    clock_t end = clock();
    cout<<endl<<"TIME TAKEN: "<<(float(end-begin)/CLOCKS_PER_SEC)<<" seconds"<<endl;
    return 1;

}
