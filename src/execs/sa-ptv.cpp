#include "std_include.h"

#include "calibration.h"
#include "refocusing.h"
#include "rendering.h"
#include "pLoc.h"
#include "tracking.h"
#include "tools.h"
#include "parse_settings.h"
#include "optimization.h"
#include "typedefs.h"
#include "batchProc.h"
#include "visualize.h"

#include "cuda_lib.h"
#include "cuda_profiler_api.h"

using namespace cv;
using namespace std;

DEFINE_bool(live, false, "live refocusing");
DEFINE_bool(find, false, "find particles");
DEFINE_bool(track, false, "track particles");
DEFINE_bool(fhelp, false, "show config file options");


int main(int argc, char** argv) {

    google::ParseCommandLineFlags(&argc, &argv, true);
    init(argc, argv);
    
    /*
    int batch = 0;

    if (batch) {

        batchFind job(argv[1]);
        job.run();

    } else {
    
    refocus_settings settings;
    parse_refocus_settings(string(argv[1]), settings, FLAGS_fhelp);

    if (FLAGS_live) {

        saRefocus refocus(settings);
        refocus.GPUliveView();

    } else if (FLAGS_find) {

        saRefocus refocus(settings);
        refocus.initializeGPU();
        localizer_settings s2;
        s2.window = 2; s2.thresh = 40.0; s2.zmethod = 2;
        s2.zmin = -10;
        s2.zmax = 110.0; //40
        s2.dz = 0.05;
        s2.show_particles = 0;
        pLocalize localizer(s2, refocus, settings);

        localizer.find_particles_all_frames();
        //localizer.write_all_particles(settings.images_path);
        //localizer.write_all_particles("../temp/");
        localizer.write_all_particles_to_file("../temp/particles/particles_idz20_zm2.txt");

    } else if (FLAGS_track) {

        pTracking track(string(argv[1]), atof(argv[2]), atof(argv[3]));
        track.track_all();
        //track.plot_complete_paths();
        track.write_quiver_data();

    }

    }
    */

    double f = 8.0;
    int xv = 1000; int yv = 1000; int zv = 500; int particles = 1000;
    Scene scn;
    scn.create(xv/f, yv/f, zv/f, 1);
    scn.seedParticles(particles, 0.75);
    scn.renderVolume(xv, yv, zv);
    scn.setRefractiveGeom(-100, 1.0, 1.5, 1.33, 5);

    string path = "/home/ab9/projects/scenes/test/";
    stringstream ss;
    ss<<path<<"scene_"<<xv<<"_"<<yv<<"_"<<zv<<"_"<<particles<<"_ref_1.obj";
    string filename = ss.str();
 
    saveScene(filename, scn);
 
    // loadScene(filename, scn);

    // Camera cam;
    // double cf = 35.0; // [mm]
    // cam.init(cf*1200/4.8, xv, yv, 1);
    // cam.setScene(scn);
    
    // benchmark bm;
    // double d = 1000;
    // vector<double> th, q;

    // double ang = 30;

    // saRefocus ref;
    // ref.setRefractive(1, -100, 1.0, 1.5, 1.33, 5);
    // ref.setHF(1);
    // addCams(scn, cam, ang, d, f, ref);
    // ref.initializeGPU();

    // refocus_settings settings;
    // localizer_settings s2;
    // s2.window = 2; s2.thresh = 30.0; s2.zmethod = 1;
    // s2.zmin = -zv*0.5*1.1/8;
    // s2.zmax = zv*0.5*1.1/8;
    // s2.dz = 0.1;
    // s2.show_particles = 0;
    // pLocalize localizer(s2, ref, settings);
    // localizer.find_particles_all_frames();

    // ref.GPUliveView();

    // bm.benchmarkSA(scn, ref);
    // LOG(INFO)<<bm.calcQ(60.0, 0, 0);
    
    return 1;

}
