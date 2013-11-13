#include "std_include.h"

#include "calibration.h"
#include "refocusing.h"
#include "pLoc.h"
#include "tracking.h"
#include "tools.h"
#include "optimization.h"

#include "cuda_lib.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
        
    pTracking track(argv[1], 2.0, 2.0);
    track.read_points();
    track.track_all();
    //track.plot_complete_paths();
    track.write_quiver_data(0, "../matlab/quiver.txt");

    return 0;

}
