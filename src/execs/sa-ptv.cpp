#include "std_include.h"

#include "calibration.h"
#include "refocusing.h"
#include "pLoc.h"
#include "tracking.h"
#include "tools.h"
#include "optimization.h"
#include "typedefs.h"

#include "cuda_lib.h"
#include "cuda_profiler_api.h"

#include "gperftools/profiler.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

    refocus_settings settings;
    settings.gpu = 1;
    settings.ref = 1;
    settings.mult = 0;
    settings.mult_exp = 1/9.0;
    settings.corner_method = 1;
    settings.calib_file_path = string(argv[1]);
    settings.images_path = string(argv[2]);
    settings.mtiff = 1;
    settings.start_frame = 30;
    settings.end_frame = 35;
    settings.upload_frame = -1;
    
    //string particle_file("../temp/particles_run2_t70.txt");
    
    int window = 2;
    double thresh = 100.0;
    int cluster = 8;
    string date("11-08");
    string run("4");

    stringstream s;
    s<<"../particle_files/date"; s<<date<<"_"; s<<"run"<<run<<"_"; s<<"f"<<settings.start_frame<<"to"<<settings.end_frame<<"_"; s<<"w"<<window<<"_"; s<<"t"<<thresh<<"_"<<"c"<<cluster<<".txt";
    string particle_file = s.str();
    cout<<particle_file<<endl;

    int find = 0;
    int track = 0;
    int live = 1;
    
    if (find==1 || live ==1) {

        saRefocus refocus(settings);

        if (live) {
            refocus.GPUliveView();
        }

        if (find) {
            refocus.initializeGPU();
            
            localizer_settings s2;

            s2.window = window;
            s2.cluster_size = cluster;
            s2.zmin = 20.0; //-20
            s2.zmax = 60.0; //40
            s2.dz = 0.1;
            s2.thresh = thresh; //90.0; //100.0
            pLocalize localizer(s2, refocus);
            
            //localizer.z_resolution();
            //localizer.crop_focus();
            
            //localizer.run();
            //localizer.write_particles_to_file("../matlab/particles_found.txt");

            localizer.find_particles_all_frames();
            localizer.write_all_particles_to_file(particle_file);        
        }
        
    }
    
    if (track) {
        pTracking track(particle_file, 3.0, 3.0);
        track.read_points();
        track.track_all();
        track.plot_complete_paths();
        track.write_quiver_data(0, "../matlab/quiver.txt");
    }   

    return 1;

}
