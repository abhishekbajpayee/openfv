#include "std_include.h"

#include "calibration.h"
#include "refocusing.h"
#include "pLoc.h"
#include "tracking.h"
#include "tools.h"

using namespace cv;
using namespace std;

DEFINE_string(path, "../temp/", "Calibration path");
DEFINE_int32(hgrid, 5, "Horizontal grid size");
DEFINE_int32(vgrid, 5, "Horizontal grid size");
DEFINE_double(gridsize, 5, "Physical grid size");
DEFINE_bool(ref, false, "Refractive flag");
DEFINE_bool(mtiff, false, "Multipage tiff flag");

int main(int argc, char** argv) {

    // Parsing flags
    google::ParseCommandLineFlags(&argc, &argv, true);
    init_logging(argc, argv);

    Size grid_size = Size(FLAGS_hgrid, FLAGS_vgrid); // Format (horizontal_corners, vertical_corners)

    // Uses dummy mode
    multiCamCalibration calibration(FLAGS_path, grid_size, FLAGS_gridsize, FLAGS_ref, 1, FLAGS_mtiff);
    calibration.run();

    cout<<"DONE!"<<endl;
    
    return 1;

}
