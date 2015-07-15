#include "std_include.h"

#include "calibration.h"
#include "refocusing.h"
#include "pLoc.h"
#include "tracking.h"
#include "tools.h"
#include "parse_settings.h"

using namespace cv;
using namespace std;

DEFINE_bool(live, false, "live refocusing");
DEFINE_bool(fhelp, false, "show config file options");

int main(int argc, char** argv) {

    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr=1;

    refocus_settings settings;
    parse_refocus_settings(string(argv[1]), settings, FLAGS_fhelp);
    saRefocus refocus(settings); 
    
    refocus.initializeGPU();
    // refocus.dump_stack(settings.save_path, settings.zmin, settings.zmax, settings.dz, settings.thresh, "tif");

    double t1 = omp_get_wtime();

    vector<Mat> stack;
    refocus.return_stack(-55, 55, 0.1, 30, 0, stack);

    double t2 = omp_get_wtime();

    double t = t2-t1;
    LOG(INFO)<<"Time: "<<t;
    LOG(INFO)<<"Speed Improvement: "<<96.0*60/t<<"x";

    return 1;

}
