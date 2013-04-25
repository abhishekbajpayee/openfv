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
    
    vector<Point2f> points;
    vector<particle2d> particles;
    Mat result;
    double thresh = 50.0;

    particle2d particle;
    vector<particle2d> particles3D;
    
    pLocalize localizer(2);

    for (float i=-50.0; i<-40.0; i += 0.1) {

        refocus.GPUrefocus(i, thresh, 0);
        Mat image = refocus.result;
        localizer.find_particles(image, points);
        localizer.refine_subpixel(image, points, particles);
        localizer.draw_points(image, result, points);
        //imshow("result", result);
        //waitKey(0);
        for (int j=0; j<particles.size(); j++) {
            particle.x = particles[j].x/refocus.scale();
            particle.y = particles[j].y/refocus.scale();
            particle.z = i;
            particle.I = particles[j].I;
            particles3D.push_back(particle);
        }
        cout<<i<<endl;
        points.clear();
        particles.clear();

    }

    ofstream file;
    file.open("matlab/particles.txt");
    for (int i=0; i<particles3D.size(); i++) {
        file<<particles3D[i].x<<"\t"<<particles3D[i].y<<"\t"<<particles3D[i].z<<endl;
    }
    file.close();

    vector< vector<particle2d> > clusters;

    while(particles3D.size()) {

        double x = particles3D[0].x;
        double y = particles3D[0].y;
        double z = particles3D[0].z;

        double xydist, zdist;
        double xythresh = 0.5;
        double zthresh = 1.0;

        vector<int> used;

        vector<particle2d> single_particle;
        single_particle.push_back(particles3D[0]);
        for (int i=1; i<particles3D.size(); i++) {
            xydist = sqrt( pow((particles3D[i].x-x), 2) + pow((particles3D[i].y-y), 2) );
            zdist = abs( particles3D[i].z -z );
            if (xydist<xythresh && zdist<zthresh) {
                single_particle.push_back(particles3D[i]);
                used.push_back(i);
            }
        }
        
        // The tricky delete loop
        for (int i=used.size()-1; i>=0; i--) {
            particles3D.erase(particles3D.begin()+used[i]);
        }
        particles3D.erase(particles3D.begin());
        
        clusters.push_back(single_particle);
        single_particle.clear();
        used.clear();

    }

    cout<<clusters.size()<<" clusters found."<<endl;
    for (int i=clusters.size(); i>=0; i--) {
        if (clusters[i].size()<=5) {
            clusters.erase(clusters.begin()+i);
        }
    }

    clock_t end = clock();
    cout<<endl<<"TIME TAKEN: "<<(float(end-begin)/CLOCKS_PER_SEC)<<" seconds"<<endl;
    return 1;

}
