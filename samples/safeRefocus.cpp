// Written by Barry Scharfman, Decemeber 2014
// safeRefocus.cpp (Synthetic Aperture Feature Extraction)
// Based on refocus.cpp

#include "std_include.h"

#include "calibration.h"
#include "refocusing.h"
#include "pLoc.h"
#include "tracking.h"
#include "tools.h"
#include "parse_settings.h"
#include "parse_safe_settings.h"

#include "safeRefocusing.h"

using namespace cv;
using namespace std;

//DEFINE_bool(live, false, "live refocusing");
DEFINE_bool(fhelp, false, "show config file options");

int main(int argc, char** argv) {

    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr=1;

    // parse SA refocus settings
    refocus_settings settings;
    parse_refocus_settings(string(argv[1]), settings, FLAGS_fhelp);
    LOG(INFO)<<"Finished parsing refocusing settings"<<endl;    
    
    // parse SAFE refocus settings
    safe_refocus_settings safe_settings;
    parse_safe_settings(string(argv[2]), safe_settings, FLAGS_fhelp);
    // gKerWid and gKerHt must both be odd!
    LOG(INFO)<<"Finished parsing SAFE refocusing settings"<<endl; 
   
    // Initialize safeRefocus object named safeR
    safeRefocus safeR(settings, safe_settings);
    LOG(INFO)<<"Finished initializing safeRefocus object"<<endl;     

    // Preprocess images for SAFE   
    safeR.safe_preprocess_all();
    LOG(INFO)<<"Finished preprocessing all raw images for SAFE"<<endl; 
    
    // Initialize GPU
    safeR.initializeGPU();
    LOG(INFO)<<"Finished initializing GPU"<<endl; 

    // // Refocus preprocessed images and save refocused image stack
    //safeR.dump_stack(settings.save_path, settings.zmin, settings.zmax, settings.dz, settings.thresh, settings.type);

    // Refocus and extract 3D sphere data
    safeR.perform_SAFE(settings.save_path, settings.zmin, settings.zmax, settings.dz, settings.thresh);
    LOG(INFO)<<"Finished performing SAFE"<<endl; 

    return 1;

}
