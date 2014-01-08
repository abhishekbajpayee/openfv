#include "std_include.h"

#include "calibration.h"
#include "refocusing.h"
#include "pLoc.h"
#include "tracking.h"
#include "tools.h"
#include "optimization.h"
#include "typedefs.h"
#include "batchProc.h"

#include "cuda_lib.h"
#include "cuda_profiler_api.h"

#include "gperftools/profiler.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

    int batch = 0;

    if (batch) {

        batchFind job(argv[1]);
        job.run();

    } else {
    
    refocus_settings settings;
    settings.gpu = 1; settings.ref = 1; settings.mult = 0;
    settings.corner_method = 1;
    settings.calib_file_path = string(argv[1]);
    settings.images_path = string(argv[2]);
    settings.mtiff = 1;
    settings.all_frames = 0; settings.start_frame = 78; settings.end_frame = 81;
    settings.upload_frame = -1;
    settings.preprocess = 1;

    int task = 2;

    //saRefocus refocus(settings);
    //gpu::DeviceInfo gpuDevice(gpu::getDevice());
    //cout<<"Free Memory: "<<(gpuDevice.freeMemory()/pow(1024.0,2))<<" MB"<<endl<<endl;

    switch (task) {
        
    case 1: {

        saRefocus refocus(settings);
        refocus.GPUliveView();
        break;

    }

    case 2: {

        saRefocus refocus(settings);
        refocus.initializeGPU();
        localizer_settings s2;
        s2.window = 2; s2.thresh = 50.0; s2.zmethod = 2;
        s2.zmin = 0; //-20
        s2.zmax = 50.0; //40
        s2.dz = 0.1;
        s2.show_particles = 0;
        pLocalize localizer(s2, refocus, settings);

        localizer.find_particles_all_frames();
        localizer.write_all_particles_to_file(string(argv[2]));
        break;

    }

    case 3: {

        pTracking track(string(argv[1]), atof(argv[2]), atof(argv[3]));
        track.track_all();
        //track.plot_complete_paths();
        track.write_quiver_data();

    }

    }

    }

    return 1;

}
