#include "std_include.h"

#include "refocusing.h"

using namespace cv;
using namespace std;

void parse_refocus_settings(string filename, refocus_settings &settings, bool h) {

    namespace po = boost::program_options;

    po::options_description desc("Allowed config file options");
    desc.add_options()
        ("refractive", po::value<int>(), "ON to use refractive refocusing")
        ("use_gpu", po::value<int>(), "ON to use GPU")
        ("mult", po::value<int>(), "ON to use multiplicative method")
        ("mult_exp", po::value<double>(), "Multiplicative method exponent")
        ("hf_method", po::value<int>(), "ON to use HF method")
        ("mtiff", po::value<int>(), "ON if data is in multipage tiff files")
        ("all_frames", po::value<int>(), "ON to process all frames in a multipage tiff file")
        ("start_frame", po::value<int>(), "first frame in range of frames to process")
        ("end_frame", po::value<int>(), "last frame in range of frames to process")
        ("preprocess", po::value<int>(), "ON to use preprocessing")
        ("preprocess_file", po::value<string>(), "preprocess config file to use")
        ("upload_frame", po::value<int>()->default_value(-1), "frame to upload to GPU (-1 uploads all frames)")
        ("calib_file", po::value<string>(), "calibration file to use")
        ("data_path", po::value<string>(), "path where data is located")
        ("dump_stack", po::value<int>(), "ON to save stack to path")
        ("zmin", po::value<double>(), "zmin")
        ("zmax", po::value<double>(), "zmax")
        ("dz", po::value<double>(), "dz")
        ("thresh", po::value<double>(), "threshold level")
        ("save_path", po::value<string>(), "path where results are saved");

    if (h) {
        cout<<desc;
        exit(1);
    }

    po::variables_map vm;
    po::store(po::parse_config_file<char>(filename.c_str(), desc), vm);
    po::notify(vm);

    settings.ref = vm["refractive"].as<int>();
    settings.gpu = vm["use_gpu"].as<int>();
    settings.hf_method = vm["hf_method"].as<int>();
    settings.mtiff = vm["mtiff"].as<int>();
    settings.mult = vm["mult"].as<int>();
    if (settings.mult)
        settings.mult_exp = vm["mult_exp"].as<double>();


    settings.preprocess = vm["preprocess"].as<int>();
    if (settings.preprocess)
        settings.preprocess_file = vm["preprocess_file"].as<string>();
    
    settings.all_frames = vm["all_frames"].as<int>();
    if (!settings.all_frames) {
        settings.start_frame = vm["start_frame"].as<int>();
        settings.end_frame = vm["end_frame"].as<int>();
    }
    settings.upload_frame = vm["upload_frame"].as<int>();

    settings.calib_file_path = vm["calib_file"].as<string>();
    settings.images_path = vm["data_path"].as<string>();

    if (vm["dump_stack"].as<int>()) {
        settings.zmin = vm["zmin"].as<double>();
        settings.zmax = vm["zmax"].as<double>();
        settings.dz = vm["dz"].as<double>();
        settings.thresh = vm["thresh"].as<double>();
        settings.save_path = vm["save_path"].as<string>();
    }

}
