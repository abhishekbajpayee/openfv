#include "std_include.h"

#include "calibration.h"
#include "refocusing.h"
#include "pLoc.h"
#include "tracking.h"
#include "tools.h"

using namespace cv;
using namespace std;

//TODO: CHANGE SO IT ASSUMES CALIBRATION HAS BEEN PERFORMED
int main(int argc, char** argv) {

    // Camera Calibration Section
    
    if (argc != 6) {
        cout<<endl<<"Invalid number of arguments!"<<endl;
        cout<<"Usage: liveRefocus [calibration path] [refocusing path] [frame to upload (-1 for all)] [mult/add flag (0 for add)] [use CPU or GPU (1 for GPU)]"<<endl<<endl;
        return 0;
    }

    string calib_path(argv[1]); // Folder where calibration images lie
    Size grid_size = Size(6,5); // Format (horizontal_corners, vertical_corners)
    double grid_size_phys = 5;  // in [mm]
    multiCamCalibration calibration(calib_path, grid_size, grid_size_phys, 0);
    calibration.run();

    stringstream sframe(argv[3]);
    int frame;
    sframe>>frame;

    stringstream smult(argv[4]);
    int mult;
    smult>>mult;
    
    stringstream smethod(argv[5]);
    int method;
    smethod>>method;

    double mult_exp = 1.0/9.0;

    string refoc_path(argv[2]);
    saRefocus refocus(calibration.refocusing_params(), frame, mult, mult_exp);
    refocus.read_imgs(refoc_path);

    if (method) {
        refocus.GPUliveView();
    } else {
        refocus.CPUliveView();
    }

    cout<<"DONE!"<<endl;

    return 1;

}
