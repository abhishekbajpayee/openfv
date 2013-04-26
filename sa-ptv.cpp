#include "std_include.h"
//#include "lib_include.h"

#include "calibration.h"
#include "refocusing.h"
#include "pLoc.h"
#include "tools.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    
    // Camera Calibration Section
    clock_t begin = clock();

    string calib_path("../experiment/calibration_rect/"); // Folder where calibration images lie
    Size grid_size = Size(6,5); // Format (horizontal_corners, vertical_corners)
    double grid_size_phys = 5;  // in [mm]
    
    multiCamCalibration calibration(calib_path, grid_size, grid_size_phys);
    calibration.run();

    string refoc_path("../experiment/piv_sim_500/");

    saRefocus refocus(calibration.refocusing_params());
    refocus.read_imgs(refoc_path);
    //refocus.GPUliveView(); 
    
    refocus.initializeGPU();

    int window = 2;
    int cluster_size = 10;
    double zmin = -20.0; //-20
    double zmax = 40.0; //40
    double dz = 0.1;
    double thresh = 100.0;
    pLocalize localizer(window, zmin, zmax, dz, thresh, cluster_size, refocus);
    localizer.run();

    vector<Point3f> particles = localizer.detected_particles();

    cout<<particles.size()<<" particles found."<<endl;

    ofstream file;
    file.open("matlab/particles.txt");
    for (int i=0; i<particles.size(); i++) {
        file<<particles[i].x<<"\t";
        file<<particles[i].y<<"\t";
        file<<particles[i].z<<endl;
    }
    file.close();
    
    clock_t end = clock();
    cout<<endl<<"TIME TAKEN: "<<(float(end-begin)/CLOCKS_PER_SEC)<<" seconds"<<endl;
    return 1;

}
