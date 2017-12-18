#include "openfv.h"

using namespace cv;
using namespace std;

DEFINE_bool(fhelp, false, "show config file options");
DEFINE_string(config_file, "", "config file path");

int main(int argc, char** argv) {

    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr=1;

    if (FLAGS_config_file.empty())
        LOG(FATAL) << "The config_file parameter is required! Please pass it as: \nsa_reconstruct --config_file <filename>";
    if (!boost::filesystem::exists(FLAGS_config_file))
        LOG(FATAL) << "Seems like config_file " << FLAGS_config_file << " does not exist!";

    refocus_settings ref_settings;
    reconstruction_settings rec_settings;
    parse_refocus_settings(FLAGS_config_file, ref_settings, FLAGS_fhelp);
    parse_reconstruction_settings(FLAGS_config_file, rec_settings, FLAGS_fhelp);
    
    if (rec_settings.zmax <= rec_settings.zmin)
        LOG(FATAL) << "The specified zmax must be greater than / equal to zmin!";

    if (rec_settings.thresh < 0)
        LOG(FATAL) << "The specified threshold level (thresh) cannot be negative!";

    if (ref_settings.mult)
        LOG(INFO) << "Multiplicative reconstruction is ON. Threshold level (thresh) will be ignored...";

    if (ref_settings.minlos)
        LOG(INFO) << "minLOS reconstruction is ON. Threshold level (thresh) will be ignored...";        

    saRefocus refocus(ref_settings);

    double dz;
    double scale = refocus.scale();
    if (rec_settings.manual_dz)
        dz = rec_settings.dz;
    else    
        dz = 1/scale;

    refocus.dump_stack(rec_settings.save_path, rec_settings.zmin, rec_settings.zmax, dz, rec_settings.thresh, "tif");

    refocus.write_piv_settings(rec_settings.save_path, rec_settings.zmin, rec_settings.zmax, dz, rec_settings.thresh);

    return 1;

}
