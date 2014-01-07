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
    settings.gpu = 1;
    settings.ref = 1;
    settings.mult = 0;
    settings.mult_exp = 1/9.0;
    settings.corner_method = 0;
    settings.calib_file_path = string(argv[1]);
    settings.images_path = string(argv[2]);
    settings.mtiff = 0;
    settings.all_frames = 1;
    settings.start_frame = 108;
    settings.end_frame = 112;
    settings.upload_frame = 0;
    
    //string particle_file("../temp/particles_run2_t70.txt");
    
    int window = 2;
    double thresh = 30.0;
    int zmeth = 2;

    stringstream s;
    s<<string(argv[2])<<"particles/";
    if (settings.all_frames) {
        s<<"f_all_";
    } else {
        s<<"f"<<settings.start_frame<<"to"<<settings.end_frame<<"_";
    }
    s<<"w"<<window<<"_";
    s<<"t"<<thresh<<"_";
    s<<"zm"<<zmeth<<".txt";
    string particle_file = s.str();
    cout<<particle_file<<endl;

    int find = 1;
    int live = !find;

    saRefocus refocus(settings);

    gpu::DeviceInfo gpuDevice(gpu::getDevice());
    cout<<"Free Memory: "<<(gpuDevice.freeMemory()/pow(1024.0,2))<<" MB"<<endl<<endl;

    if (live) {
        refocus.GPUliveView();
        /*
        vector<int> comparams;
        comparams.push_back(CV_IMWRITE_JPEG_QUALITY);
        comparams.push_back(100);

        refocus.initializeGPU();
        float zmin=-5.0;
        float zmax=-4.9;

        double start = omp_get_wtime();
        for (float i=zmin; i<=zmax; i += 0.1) {
            refocus.refocus(i, thresh, 0);
            Mat image = refocus.result;
            stringstream fn;
            fn<<"../../stack_orig/";
            fn<<(i-zmin)<<".jpg";
            //imwrite(fn.str(), image, comparams);
            cout<<i<<endl;
        }
        cout<<omp_get_wtime()-start<<endl;
        */
    }

    if (find) {

        refocus.initializeGPU();
        
        localizer_settings s2;
        
        s2.window = window;
        s2.zmin = -5.0; //-20
        s2.zmax = 105.0; //40
        s2.dz = 0.5;
        s2.thresh = thresh; //90.0; //100.0
        s2.zmethod = zmeth;
        pLocalize localizer(s2, refocus);

        localizer.find_particles_all_frames();
        localizer.write_all_particles_to_file(particle_file);
        
    }

    }
    return 1;

}
