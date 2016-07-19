#include "openfv.h"

using namespace cv;
using namespace std;

DEFINE_bool(live, false, "live refocusing");
DEFINE_bool(fhelp, false, "show config file options");

DEFINE_bool(dump_stack, false, "dump stack");
DEFINE_string(save_path, "", "stack save path");
DEFINE_string(config_file, "", "config file path");
DEFINE_double(zmin, -10, "zmin");
DEFINE_double(zmax, 10, "zmax");
DEFINE_double(dz, 0.1, "dz");
DEFINE_double(thresh, 0, "thresholding level");

int main(int argc, char** argv) {

    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr=1;

    refocus_settings settings;
    parse_refocus_settings(FLAGS_config_file, settings, FLAGS_fhelp);
    saRefocus refocus(settings);

    if (FLAGS_live) {
        if (settings.use_gpu) {
            refocus.GPUliveView();
        } else {
            refocus.CPUliveView();
        }
    } 

    if (FLAGS_dump_stack) {
        refocus.initializeGPU();
        refocus.dump_stack(FLAGS_save_path, FLAGS_zmin, FLAGS_zmax, FLAGS_dz, FLAGS_thresh, "tif");
    }

    return 1;

}
