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

    /*
    batchFind job(argv[1]);
    job.run();
    */
    
    refocus_settings settings;
    settings.gpu = 1;
    settings.ref = 1;
    settings.mult = 0;
    settings.mult_exp = 1/9.0;
    settings.corner_method = 0;
    settings.calib_file_path = string(argv[1]);
    settings.images_path = string(argv[2]);
    settings.mtiff = 0;
    settings.all_frames = 0;
    settings.start_frame = 108;
    settings.end_frame = 112;
    settings.upload_frame = 0;
    
    //string particle_file("../temp/particles_run2_t70.txt");
    
    int window = 2;
    double thresh = 40.0;
    int cluster = 8;

    stringstream s;
    s<<string(argv[2])<<"particles/";
    s<<"f"<<settings.start_frame<<"to"<<settings.end_frame<<"_";
    s<<"w"<<window<<"_";
    s<<"t"<<thresh<<"_";
    s<<"c"<<cluster<<".txt";
    string particle_file = s.str();
    cout<<particle_file<<endl;

    int find = 0;
    int live = !find;

    saRefocus refocus(settings);

    gpu::DeviceInfo gpuDevice(gpu::getDevice());
    cout<<"Free Memory: "<<(gpuDevice.freeMemory()/pow(1024.0,2))<<" MB"<<endl<<endl;

    if (live) {
        //refocus.GPUliveView();

        
        refocus.initializeGPU();
        float zmin=-5;
        float zmax=105;
        for (float i=zmin; i<=zmax; i += 1.0) {
            refocus.refocus(i, thresh, 0);
            Mat image = refocus.result;
            stringstream fn;
            fn<<"../../stack/";
            fn<<(i-zmin)<<".jpg";
            imwrite(fn.str(), image);
            cout<<i<<endl;
        }
        

    }

    if (find) {

        refocus.initializeGPU();
        
        localizer_settings s2;
        
        s2.window = window;
        s2.cluster_size = cluster;
        s2.zmin = -55.0; //-20
        s2.zmax = 55.0; //40
        s2.dz = 1.0;
        s2.thresh = thresh; //90.0; //100.0
        pLocalize localizer(s2, refocus);
        
        localizer.find_particles_all_frames();
        localizer.write_all_particles_to_file(particle_file);
        
    }
    
    return 1;

}
