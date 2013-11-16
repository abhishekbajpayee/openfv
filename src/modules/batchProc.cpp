#include "std_include.h"

#include "calibration.h"
#include "refocusing.h"
#include "pLoc.h"
#include "tracking.h"
#include "tools.h"
#include "optimization.h"
#include "typedefs.h"

#include "batchProc.h"

batchFind::batchFind(string path): path_(path) {}

void batchFind::run() {

    read_config_file();

    for (int i=0; i<n_; i++) {

        refocus_settings settings;
        settings.gpu = 1;
        settings.ref = 1;
        settings.mult = 0;
        settings.mult_exp = 1/9.0;
        settings.corner_method = 1;
        settings.calib_file_path = calib_paths[i];
        settings.images_path = refoc_paths[i];
        settings.mtiff = 1;
        settings.all_frames = 1;

        settings.start_frame = 5;
        settings.end_frame = 8;

        settings.upload_frame = -1;

        int window = 2;
        double thresh = threshs[i];
        int cluster = 8;

        stringstream s;
        s<<refoc_paths[i]<<"particles/";
        if (settings.all_frames) {
            s<<"f_all_";
        } else {
            s<<"f"<<settings.start_frame<<"to"<<settings.end_frame<<"_";
        }
        s<<"w"<<window<<"_";
        s<<"t"<<thresh<<"_";
        s<<"c"<<cluster<<".txt";
        string particle_file = s.str();
        cout<<particle_file<<endl;

        saRefocus refocus(settings);

        refocus.initializeGPU();

        /*
        localizer_settings s2;
        
        s2.window = window;
        s2.cluster_size = cluster;
        s2.zmin = -20.0;
        s2.zmax = 60.0;
        s2.dz = 0.1;
        s2.thresh = thresh;
        pLocalize localizer(s2, refocus);
        
        localizer.find_particles_all_frames();
        localizer.write_all_particles_to_file(particle_file);
        */

    }

}

void batchFind::read_config_file() {

    ifstream file;
    file.open(path_.c_str());

    file>>n_;

    string str;
    double t;

    // read blank line
    getline(file, str);

    for (int i=0; i<n_; i++) {

        getline(file, str);
        getline(file, str);
        calib_paths.push_back(str);
        cout<<str<<endl;

        getline(file, str);
        refoc_paths.push_back(str);
        cout<<str<<endl;

        file>>t;
        threshs.push_back(t);
        cout<<t<<endl;

        // read blank line
        getline(file, str);
        
    }

}
