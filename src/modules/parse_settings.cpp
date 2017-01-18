#include "std_include.h"

#include "refocusing.h"
#include "tools.h"

using namespace cv;
using namespace std;

void parse_refocus_settings(string filename, refocus_settings &settings, bool h) {

    namespace po = boost::program_options;

    po::options_description desc("Allowed config file options");
    desc.add_options()
        ("use_gpu", po::value<int>()->default_value(0), "ON to use GPU")
        ("mult", po::value<int>()->default_value(0), "ON to use multiplicative method")
        ("mult_exp", po::value<double>()->default_value(.25), "Multiplicative method exponent")
        ("hf_method", po::value<int>()->default_value(0), "ON to use HF method")
        ("mtiff", po::value<int>()->default_value(0), "ON if data is in multipage tiff files")
        //("all_frames", po::value<int>()->default_value(1), "ON to process all frames in a multipage tiff file")
        //("start_frame", po::value<int>()->default_value(0), "first frame in range of frames to process")
        ("frames", po::value<string>()->default_value(""), "Array of values in format start, end, skip")
        //("end_frame", po::value<int>(), "last frame in range of frames to process")
        //("upload_frame", po::value<int>()->default_value(-1), "frame to upload to GPU (-1 uploads all frames)")
        ("calib_file_path", po::value<string>()->default_value(""), "calibration file to use")
        ("images_path", po::value<string>()->default_value(""), "path where data is located")
        // ("dump_stack", po::value<int>()->default_value(0), "ON to save stack to path")
        // ("zmin", po::value<double>()->default_value(0), "zmin")
        // ("zmax", po::value<double>()->default_value(0), "zmax")
        // ("dz", po::value<double>()->default_value(0), "dz")
        // ("thresh", po::value<double>()->default_value(0), "threshold level")
        // ("save_path", po::value<string>()->default_value(""), "path where results are saved")
        ;

    if (h) {
        cout<<desc;
        exit(1);
    }

    po::variables_map vm;
    po::store(po::parse_config_file<char>(filename.c_str(), desc), vm);
    po::notify(vm);

    settings.use_gpu = vm["use_gpu"].as<int>();
    settings.hf_method = vm["hf_method"].as<int>();
    settings.mtiff = vm["mtiff"].as<int>();
    settings.mult = vm["mult"].as<int>();
    if (settings.mult)
        settings.mult_exp = vm["mult_exp"].as<double>();

    vector<int> frames;
    stringstream frames_stream(vm["frames"].as<string>());
    int i;
    while (frames_stream >> i) {
        frames.push_back(i);
        
        if(frames_stream.peek() == ',' || frames_stream.peek() == ' ') {
            frames_stream.ignore();
        }
    }
    if (frames.size() == 0) {
        settings.all_frames = 1;
        settings.skip = 0;
    } else if (frames.size() == 1) {
        settings.start_frame = frames.at(0);
        settings.end_frame = frames.at(0);
        settings.skip = 0;
        settings.all_frames = 0;
    } else if (frames.size() == 2) {
        settings.start_frame = frames.at(0);
        settings.end_frame = frames.at(1);
        settings.skip = 0;
        settings.all_frames = 0;
    } else if (frames.size() >= 3) {
        settings.start_frame = frames.at(0);
        settings.end_frame = frames.at(1);
        settings.skip = frames.at(2);
        settings.all_frames = 0;
    }
   if (settings.start_frame<0) {
            LOG(FATAL)<<"Can't have starting frame less than 0. Terminating..."<<endl;
        }
   
    // settings.all_frames = vm["all_frames"].as<int>();
    // if (!settings.all_frames) {
    //     settings.start_frame = vm["start_frame"].as<int>();
    //     settings.end_frame = vm["end_frame"].as<int>();
    // }
    // settings.upload_frame = vm["upload_frame"].as<int>();

    boost::filesystem::path calibP(filename);
    boost::filesystem::path imgsP(filename);

    calibP.remove_leaf() /= vm["calib_file_path"].as<string>();
    imgsP.remove_leaf() /= vm["images_path"].as<string>();

    
    settings.calib_file_path = calibP.string();
    if (settings.calib_file_path.empty()) {
        LOG(FATAL)<<"calib_file is a REQUIRED Variable";
    }
    // else if (!dirExists(settings.calib_file_path)) {
    //    LOG(FATAL)<<"Calibration File Path does not exist!";
    // }
    
     
    settings.images_path = imgsP.string();
    if(settings.images_path.empty()) {
        LOG(FATAL)<<"data_path is a REQUIRED Variable";
    }
    //else if (!dirExists(settings.images_path)) {
    //    LOG(FATAL)<<"Images Files Path does not exist!";
    // } 

  
}

void parse_calibration_settings(string filename, calibration_settings &settings, bool h) {

    namespace po = boost::program_options;

    po::options_description desc("Allowed config file options");
    desc.add_options()
        ("images_path", po::value<string>()->default_value(""), "path where data is located")
        ("refractive", po::value<int>()->default_value(0), "ON if calibration data is refractive")
        ("mtiff", po::value<int>()->default_value(0), "ON if data is in multipage tiff files")
        ("mp4", po::value<int>()->default_value(0), "ON if data is in mp4 files")
        ("hgrid", po::value<int>()->default_value(5), "Horizontal number of corners in the grid")
        ("vgrid", po::value<int>()->default_value(5), "Vertical number of corners in the grid")
        ("grid_size_phys", po::value<double>()->default_value(5), "Physical size of grid in [mm]")
        ("skip", po::value<int>()->default_value(1), "Number of frames to skip (used mostly when mtiff on)")
        ("frames", po::value<string>()->default_value(""), "Array of values in format start, end, skip")
        ;

    if (h) {
        cout<<desc;
        exit(1);
    }

    po::variables_map vm;
    po::store(po::parse_config_file<char>(filename.c_str(), desc), vm);
    po::notify(vm);

    settings.refractive = vm["refractive"].as<int>();
    settings.mtiff = vm["mtiff"].as<int>();
    settings.skip = vm["skip"].as<int>();
    settings.mp4 = vm["mp4"].as<int>();
    if (settings.mp4) {
        vector<int> frames;
        stringstream frames_stream(vm["frames"].as<string>());
        int i;
        while (frames_stream >> i) {
            frames.push_back(i);
            
            if(frames_stream.peek() == ',' || frames_stream.peek() == ' ') {
                frames_stream.ignore();
            }
        }
        if (frames.size() == 3) {
            settings.start_frame = frames.at(0);
            settings.end_frame = frames.at(1);
            settings.skip = frames.at(2);
        } else {
            LOG(FATAL)<<"frames expects 3 comma or space separated values";
        }
    }
    if (settings.start_frame<0) {
        LOG(FATAL)<<"Can't have starting frame less than 0. Terminating..."<<endl;
    }
   
    boost::filesystem::path imgsP(filename);
    imgsP.remove_leaf() /= vm["images_path"].as<string>();
     
    settings.images_path = imgsP.string();
    if(settings.images_path.empty()) {
        LOG(FATAL)<<"images_path is a REQUIRED variable";
    }
  
}
