#include "std_include.h"

#include "calibration.h"
#include "refocusing.h"
#include "pLoc.h"
#include "tracking.h"
#include "tools.h"
#include "optimization.h"

#include "cuda_lib.h"
#include "cuda_profiler_api.h"

//#include "gperftools/profiler.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
        
    int frame_list[] = {38,39,40,41};
    vector<int> frames(frame_list, frame_list + sizeof(frame_list) / sizeof(int));
    
    int mult = 0;
    double mult_exp = 1.0/9.0;

    int frame = -1;
    saRefocus refocus(string(argv[1]), frames, frame, mult, mult_exp, 1); // last argument is CORNER_FLAG
    //refocus.read_imgs(string(argv[2]));
    refocus.read_imgs_mtiff(string(argv[2]));

    //ProfilerStart("profile");
    //refocus.GPUrefocus_ref_corner(0, 0, 0, 0);
    //ProfilerStop();
    
    string particle_file("../temp/particles.txt");
    
    int find = 0;
    int track = 0;
    int live = 1;

    if (live) {
        refocus.GPUliveView();
    } 

    if (find) {
        refocus.initializeGPU();
        
        int window = 1;
        int cluster_size = 10;
        double zmin = 20.0; //-20
        double zmax = 60.0; //40
        double dz = 0.1;
        double thresh = 0.11; //90.0; //100.0
        pLocalize localizer(window, zmin, zmax, dz, thresh, cluster_size, refocus);

        //localizer.z_resolution();
        //localizer.crop_focus();
        
        //localizer.run();
        //localizer.write_particles_to_file("../matlab/particles_found.txt");
        
        localizer.find_particles_all_frames();
        localizer.write_all_particles_to_file(particle_file);        
    }
    
    if (track) {
        pTracking track(particle_file);
        track.read_points();
        track.track_all();
    }

    //cout<<endl<<"TIME TAKEN: "<<(omp_get_wtime()-wall_timer)<<" seconds"<<endl;    

    return 1;

}
