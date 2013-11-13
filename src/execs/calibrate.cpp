#include "std_include.h"

#include "calibration.h"
#include "refocusing.h"
#include "pLoc.h"
#include "tracking.h"
#include "tools.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

    // Camera Calibration Section
    
    if (argc != 5) {
        cout<<endl<<"Invalid number of arguments!"<<endl;
        cout<<"Usage: calibrate [calibration_path] [horizontal grid size] [vertical grid size] [physical grid size (mm)]"<<endl<<endl;
    }

    string calib_path(argv[1]); // Folder where calibration images lie
    
    stringstream xg(argv[2]);
    int xsize;
    xg>>xsize;    
    stringstream yg(argv[3]);
    int ysize;
    yg>>ysize;
    Size grid_size = Size(xsize,ysize); // Format (horizontal_corners, vertical_corners)

    stringstream ps(argv[4]);
    double grid_size_phys;  // in [mm]
    ps>>grid_size_phys;

    // Uses dummy mode
    multiCamCalibration calibration(calib_path, grid_size, grid_size_phys, 0, 1, 1);
    calibration.run();

    cout<<"DONE!"<<endl;
    
    return 1;

}
