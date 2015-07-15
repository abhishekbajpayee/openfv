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
#include "piv.h"

#include "cuda_lib.h"
#include "cuda_profiler_api.h"

using namespace cv;
using namespace std;

DEFINE_bool(live, false, "live refocusing");
DEFINE_bool(find, false, "find particles");
DEFINE_bool(track, false, "track particles");
DEFINE_bool(fhelp, false, "show config file options");
DEFINE_bool(cpiv, false, "test c++ piv code");

DEFINE_bool(save, false, "save scene");
DEFINE_bool(sp, false, "show particles");
DEFINE_bool(piv, false, "PIV mode");
DEFINE_bool(visualizeField, false, "play movie of velocity field");
DEFINE_bool(renderTest, false, "render volume for PIV in matlab");
DEFINE_bool(runPIV, false, "run PIV in C++");
DEFINE_bool(saveRefStack, false, "save reference stack to run PIV in matlab");
DEFINE_bool(param, false, "t param study");
DEFINE_bool(ref, false, "refractive or not");
DEFINE_bool(zpad, false, "zero padding or not");

DEFINE_int32(zm, 1, "z method");
DEFINE_int32(xv, 1, "xv");
DEFINE_int32(yv, 1, "yv");
DEFINE_int32(zv, 1, "zv");
DEFINE_int32(i, 1, "scene id");
DEFINE_int32(cs, 5, "cluster size");
DEFINE_int32(hf, 1, "HF method");
DEFINE_int32(part, 100, "particles");
DEFINE_int32(cams, 9, "num cams");
DEFINE_int32(mult, 0, "multiplicative");
DEFINE_int32(vsize, 64, "volume size for piv test");
DEFINE_int32(frames, 10, "number of frames to render movie over");
DEFINE_int32(field, 1, "velocity field to use");

DEFINE_string(pfile, "../temp/default_pfile.txt", "particle file");
DEFINE_string(rfile, "../temp/default_rfile.txt", "reference file");
DEFINE_string(scnfile, "../temp/default_scn.obj", "Scene object file");
DEFINE_string(stackpath, "../temp/stack", "stack location");

DEFINE_double(t, 0, "threshold level");
DEFINE_double(dz, 0.1, "dz");
DEFINE_double(angle, 30, "angle between cameras");
DEFINE_double(dt, 0.1, "dt for particle propagation");
DEFINE_double(e, 1, "e");
DEFINE_double(f, 0.05, "f");
DEFINE_double(rn, 3, "Rn");
DEFINE_double(rs, 3, "Rs");
DEFINE_double(multexp, 0.25, "multiplicative exponent");

int main(int argc, char** argv) {

    google::ParseCommandLineFlags(&argc, &argv, true);
    init_logging(argc, argv);
    
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
    int xv = FLAGS_xv; int yv = FLAGS_yv; int zv = FLAGS_zv; int particles = FLAGS_part;
    Scene scn;

    string path = "/home/ab9/projects/scenes/";
    stringstream ss;
    ss<<path<<"scene_"<<xv<<"_"<<yv<<"_"<<zv<<"_"<<particles;
    if (FLAGS_ref)
        ss<<"_ref_";
    else
        ss<<"_";
    ss<<FLAGS_i<<".obj";
    string filename = ss.str();

    // Create, render and save a scene
    if (FLAGS_save) {

        scn.create(xv/f, yv/f, zv/f, 1);
        if (FLAGS_ref) {
            scn.seedParticles(particles, 0.75);
            scn.renderVolume(xv, yv, zv);
            scn.setRefractiveGeom(-100, 1.0, 1.5, 1.33, 5);
        } else {
            scn.seedParticles(particles, 0.95);
            scn.renderVolume(xv, yv, zv);
        }
        saveScene(filename, scn);

    }

    if (FLAGS_find) {
 
        loadScene(FLAGS_scnfile, scn);
        fileIO fo(FLAGS_rfile);
        fo<<scn.getParticles();

        Camera cam;
        double cf = 35.0; // [mm]
        cam.init(cf*1200/4.8, xv, yv, 1);
        cam.setScene(scn);
    
        benchmark bm;
        double d = 1000;
        vector<double> th, q;

        double ang = FLAGS_angle;

        saRefocus ref;
        if (FLAGS_ref)
            ref.setRefractive(1, -100, 1.0, 1.5, 1.33, 5);
        ref.setHF(FLAGS_hf);

        // Adding cameras
        //addCams(scn, cam, ang, d, f, ref);
        double theta = ang*pi/180.0;
        ref.setF(f);

        double xy = d*sin(theta);
        double z = -d*cos(theta);
        vector< vector<Mat> > imgs;
        vector<Mat> Pmats, locations;
        for (int t=0; t<2; t++) {
            vector<Mat> views;
            for (double x = -xy; x<=xy; x += xy) {
                for (double y = -xy; y<=xy; y += xy) {
                    cam.setLocation(x, y, z);
                    if (t==0) {
                        Mat P = cam.getP();
                        Mat C = cam.getC();
                        Pmats.push_back(P);
                        locations.push_back(C);
                    }
                    Mat img = cam.render();
                    views.push_back(img);
                }
            }
            imgs.push_back(views);
            scn.propagateParticles(burgers_vortex, FLAGS_dt);
        }

        ref.addViews(imgs, Pmats, locations);
        // done adding cameras

        if (FLAGS_live) {

            ref.GPUliveView();

        } else {

            ref.initializeGPU();

            refocus_settings settings;
            localizer_settings s2;
            s2.window = 1; s2.thresh = FLAGS_t; s2.zmethod = FLAGS_zm;
            s2.zmin = -zv*0.5*1.1/8;
            s2.zmax = zv*0.5*1.1/8;
            s2.dz = FLAGS_dz;
            s2.show_particles = FLAGS_sp;
            s2.cluster_size = FLAGS_cs;
            pLocalize localizer(s2, ref, settings);
            localizer.find_particles_all_frames();
            localizer.write_all_particles_to_file(FLAGS_pfile);

        }


    }

    if (FLAGS_piv) {
    
        // Only read scenes and not generate in piv mode
        
        if (FLAGS_visualizeField) {
            
            loadScene(FLAGS_scnfile, scn);
    
            vector<int> voxels = scn.getVoxelGeom();

            Camera cam;
            double cf = 35.0; // [mm]
            cam.init(cf*1200/4.8, voxels[0], voxels[1], 1);
            cam.setScene(scn);
            
            double d = 1000;
            double t = FLAGS_t;
    
            cam.setLocation(-200, -200, -1000);
            
            vector<Mat> frames;
            for (int i=0; i<FLAGS_frames; i++) {
                Mat img = cam.render();
                frames.push_back(img);
                scn.propagateParticles(burgers_vortex, FLAGS_dt);
            }

            Movie mov(frames);

        }

        // Read a scene, render camera images, reconstruct, calculate Q
        // and then dump stack to run PIV in matlab
        if (FLAGS_renderTest) {

            loadScene(FLAGS_scnfile, scn);
            vector<int> voxels = scn.getVoxelGeom();
            LOG(INFO)<<"Voxels: "<<voxels[0]<<" x "<<voxels[1]<<" x "<<voxels[2];

            Camera cam;
            double cf = 35.0; // [mm]
            cam.init(cf*1200/4.8, voxels[0], voxels[1], 1);
            cam.setScene(scn);
            
            double d = 1000;
            double t = FLAGS_t;
    
            for (int i=0; i<2; i++) {
    
                saRefocus ref;
                ref.setF(f);
            
                double theta = FLAGS_angle*pi/180.0;
                double xy = d*sin(theta);
                double z = -d*cos(theta);     
            
                int cams = FLAGS_cams;
                double xxy, yxy;
                if (cams==4) {
                    xxy = 2*xy; yxy = 2*xy;
                } else if (cams==6) {
                    xxy = xy; yxy = 2*xy;
                } else {
                    xxy = xy; yxy = xy;
                }
    
                for (double x = -xy; x<=xy; x += xxy) {
                    for (double y = -xy; y<=xy; y += yxy) {
                        cam.setLocation(x, y, z);
                        Mat P = cam.getP();
                        Mat C = cam.getC();
                        Mat img = cam.render();
                        ref.addView(img, P, C);
                    }
                }
    
                if (FLAGS_live)
                    ref.GPUliveView();
                else
                    ref.initializeGPU();
                
                benchmark bm;
                bm.benchmarkSA(scn, ref);
    
                string path = "/home/ab9/projects/stack/piv/render/";

                string prefix;
                if (FLAGS_field==1) {
                    prefix = "ref_uniform_";
                } else if (FLAGS_field==2) {
                    prefix = "ref_vortexr_";
                } else if (FLAGS_field==3) {
                    prefix = "ref_hvortex_";
                } else {
                    LOG(FATAL)<<"Invalid velocity field!";
                }

                stringstream ss;
                ss<<path<<prefix;
                ss<<"angle"<<FLAGS_angle;
                ss<<"_cams"<<cams;
                ss<<"_mult"<<FLAGS_mult;
                if (FLAGS_mult) {
                    ss<<"_"<<FLAGS_multexp;
                } else {
                    ss<<"_"<<t;
                }
                ss<<"_"<<FLAGS_dt<<"/";
                boost::filesystem::create_directories(ss.str());

                if (FLAGS_param) {
    
                    for (t=100; t<=200; t+=5.0)
                        LOG(INFO)<<t<<"\t"<<bm.calcQ(t, FLAGS_mult, 0.25);
                    break;
    
                } else {
    
                    double q = bm.calcQ(t, FLAGS_mult, FLAGS_multexp);
                    LOG(INFO)<<"Q"<<i<<"\t"<<q;
    
                    vector<double> zs = linspace(-0.5*voxels[2]/f, 0.5*voxels[2]/f, voxels[2]);
                    ref.setMult(FLAGS_mult, FLAGS_multexp);
                    LOG(INFO)<<"Saving frame "<<i<<" to "<<ss.str();
                    ref.dump_stack_piv(ss.str(), zs[0], zs[zs.size()-1], zs[1]-zs[0], t, "tif", i, q);
                
                }
    
                if (i==0) {
                    if (FLAGS_field==1) {
                        scn.propagateParticles(test_field, FLAGS_dt);
                    } else if (FLAGS_field==2) {
                        scn.propagateParticles(burgers_vortex, FLAGS_dt);
                    } else if (FLAGS_field==3) {
                        scn.propagateParticles(hill_vortex, FLAGS_dt);
                    } else {
                        LOG(FATAL)<<"Invalid velocity field!";
                    }
                    scn.renderVolume(voxels[0], voxels[1], voxels[2]);
                }
            
            }
        
        }

        // Save reference stacks to run PIV in Matlab
        if (FLAGS_saveRefStack) {
    
            Scene scn;
            loadScene(FLAGS_scnfile, scn);
            vector<int> voxels = scn.getVoxelGeom();
            LOG(INFO)<<"Voxels: "<<voxels[0]<<" x "<<voxels[1]<<" x "<<voxels[2];

            string base_path = "/home/ab9/projects/stack/piv/render/";
            string prefix = "ren_uniform_";
            
            stringstream ss;
            ss<<base_path<<prefix<<FLAGS_dt<<"/0/refocused";
            boost::filesystem::create_directories(ss.str());
            scn.dumpStack(ss.str());

            scn.propagateParticles(test_field, FLAGS_dt);
            scn.renderVolume(voxels[0], voxels[1], voxels[2]);

            ss.str("");
            ss<<base_path<<prefix<<FLAGS_dt<<"/1/refocused";
            boost::filesystem::create_directories(ss.str());
            scn.dumpStack(ss.str());

        }

        // Read a scene, propagate particles using velocity field and
        // run PIV within C++
        if (FLAGS_runPIV) {

            piv3D piv("../temp/velocity.txt");
    
            Scene scn;
            loadScene(FLAGS_scnfile, scn);
            vector<int> voxels = scn.getVoxelGeom();
            LOG(INFO)<<"Voxels: "<<voxels[0]<<" x "<<voxels[1]<<" x "<<voxels[2];
            
            piv.add_frame(scn.getVolume());
    
            // vector<Mat> mats;
            // for (int i = 0; i < 512; i++) {
            //     Mat_<double> mat = Mat_<double>::ones(512,512);
            //     mats.push_back(mat);
            // }
    
            scn.propagateParticles(test_field, FLAGS_dt);
            scn.renderVolume(voxels[0], voxels[1], voxels[2]);
    
            piv.add_frame(scn.getVolume());
    
            piv.run(FLAGS_vsize, 0.5);
            
        }

    }

    if (FLAGS_track) {

        pTracking track(FLAGS_pfile, FLAGS_rn, FLAGS_rs);
        track.set_vars(1, FLAGS_rn, FLAGS_rs, FLAGS_e, FLAGS_f);
        track.track_all();
        //track.plot_all_paths();
        //track.write_quiver_data();

    }

    // bm.benchmarkSA(scn, ref);
    // LOG(INFO)<<bm.calcQ(60.0, 0, 0);

    return 1;

}
