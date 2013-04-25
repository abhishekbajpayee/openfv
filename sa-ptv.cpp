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
    
    vector<Point2f> points;
    vector<Point2f> points_spix;
    Mat result;
    
    pLocalize localizer(2);

    for (float i=-30.0; i<30.0; i += 0.5) {

    cout<<"z = "<<i<<endl;
    refocus.GPUrefocus(i, 50.0, 0);
    Mat image = refocus.result;
    localizer.find_particles(image, points);
    localizer.refine_subpixel(image, points, points_spix);
    localizer.draw_points(image, result, points);
    //imshow("original", image);
    imshow("result", result);
    waitKey(0);
    points.clear();
    points_spix.clear();

    }

    clock_t end = clock();
    cout<<endl<<"TIME TAKEN: "<<(float(end-begin)/CLOCKS_PER_SEC)<<" seconds"<<endl;
    return 1;

}
