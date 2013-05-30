#include "std_include.h"

#include "calibration.h"
#include "refocusing.h"
#include "pLoc.h"
#include "tracking.h"
#include "tools.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

    string particle_file(argv[1]);
    pTracking tracker(particle_file);
    tracker.read_points();
    tracker.track_all();

    cout<<"DONE!"<<endl;
    
    return 1;

}
