#include "std_include.h"

#include "calibration.h"
#include "refocusing.h"
#include "pLoc.h"
#include "tracking.h"
#include "tools.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

    char* refoc_path_c = argv[2];
    string refoc_path(refoc_path_c);

    stringstream zmins(argv[3]);
    double zmin;
    zmins>>zmin;

    stringstream zmaxs(argv[4]);
    double zmax;
    zmaxs>>zmax;

    stringstream dzs(argv[5]);
    double dz;
    dzs>>dz;

    stringstream mults(argv[6]);
    int mult;
    mults>>mult;

    stringstream mes(argv[7]);
    double mexp;
    mes>>mexp;

    stringstream threshs(argv[8]);
    double thresh;
    threshs>>thresh;

    // Camera Calibration Section
    
    string calib_path(argv[1]); // Folder where calibration images lie
    Size grid_size = Size(6,5); // Format (horizontal_corners, vertical_corners)
    double grid_size_phys = 5;  // in [mm]

    int dummy_mode = 1;
    multiCamCalibration calibration(calib_path, grid_size, grid_size_phys, dummy_mode);
    calibration.run();
    //calibration.write_calib_results_matlab();

    int frame = 0;
    saRefocus refocus(calibration.refocusing_params(), frame, mult, mexp);
    refocus.read_imgs(refoc_path);
    refocus.initializeGPU();
    
    int window = 2;
    int cluster_size = 10;
    //double thresh = 90.0; //100.0
    pLocalize localizer(window, zmin, zmax, dz, thresh, cluster_size, refocus);
    localizer.save_refocus(frame);
    
    return 1;

}
