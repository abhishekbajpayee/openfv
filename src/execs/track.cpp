#include "std_include.h"

#include "calibration.h"
#include "refocusing.h"
#include "pLoc.h"
#include "tracking.h"
#include "tools.h"
#include "optimization.h"

#include "cuda_lib.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    
    
    //double rn = atof(argv[2]);
    //double rs = atof(argv[3]);

    //pTracking track(argv[1], rn, rs);
    //track.track_frames(15, 16);
    //track.track_all();
    
    //track.write_quiver_data();
    //track.write_tracking_result();
    
    //track.plot_all_paths();
    //track.plot_complete_paths();
    //track.write_all_paths("../temp/track_movie_nn.txt");
    
    string path = string(argv[1]);

    string qpath("");
    for (int i=0; i<path.size()-4; i++) {
        qpath += path[i];
    }
    qpath += "_opt.txt";

    ofstream file;
    file.open(qpath.c_str());

    cout<<"Writing data to "<<qpath<<endl;

    for (double rn = 1; rn<=50; rn+=1) {
        cout<<rn<<"\t"; file<<rn<<"\t";
        pTracking track(argv[1], rn, rn);
        track.track_frames(15, 16);
        double res = track.sim_performance();
        cout<<res<<endl;
        file<<res<<endl;
    }

    file.close();
    
    /*
    double* params = new double[8];

    params[0] = 1.0;
    params[1] = 1.0;
    params[2] = 0.3;
    params[3] = 3.0;
    params[4] = 0.1;
    params[5] = 5.0;
    params[6] = 1.0;
    params[7] = 0.05;

    pTracking track(argv[1], params[0], params[1]);

    ceres::Problem problem;

    ceres::CostFunction* cost_function =
        new ceres::NumericDiffCostFunction<rlxTrackingError, ceres::CENTRAL, 1, 8>
        (new rlxTrackingError(track));
    
    problem.AddResidualBlock(cost_function,
                             NULL,
                             params);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    //options.max_num_iterations = pinhole_max_iterations;
    
    //int threads = omp_get_num_procs();
    //options.num_threads = threads;
    //cout<<"\nSolver using "<<threads<<" threads.\n\n";

    //options.gradient_tolerance = 1E-12;
    //options.function_tolerance = 1E-8;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout<<summary.FullReport()<<"\n";
    */
    return 1;

}
