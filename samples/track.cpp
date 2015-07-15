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
    
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr=1;
    
    double rn = atof(argv[2]);
    double rs = atof(argv[3]);
    double e = atof(argv[4]);
    double f = atof(argv[5]);

    pTracking track(argv[1], rn, rs);

    track.set_vars(1, rn, rs, e, f);

    track.track_frames(atof(argv[6]), atof(argv[7]));
    //track.track_all();
    
    //track.write_quiver_data();
    //track.write_tracking_result();
    
    //track.find_long_paths(atof(argv[8]));
    track.find_sized_paths(atof(argv[8]));
    //track.plot_long_paths();
    track.plot_sized_paths();
    //track.write_long_quiver("../temp/exp_quiver_new2.txt", atof(argv[8]));
    
    /*
    string path = string(argv[1]);

    string qpath("");
    for (int i=0; i<path.size()-4; i++) {
        qpath += path[i];
    }
    qpath += "_opt_full.txt";

    ofstream file;
    file.open(qpath.c_str());

    cout<<"Writing data to "<<qpath<<endl;

    pTracking track(argv[1], 1, 1);

    for (double rn=1; rn<30; rn+=1.0) {
        for (double e=0.1; e<=3.0; e+=0.1) {
            for (double f=0; f<=0.5; f+=0.025) {
                track.set_vars(rn, rn, e, f);
                track.track_frames(15, 16);
                double res = track.sim_performance();
                file<<rn<<"\t"<<e<<"\t"<<f<<"\t"<<res<<endl;
            }
        }
    }
    
    file.close();
    */
    /*
    double* params = new double[8];

    params[0] = 1.5;
    params[1] = 1.0;
    params[2] = 0.5;
    params[3] = 0.05;

    ceres::Problem problem;
    
    ceres::CostFunction* cost_function =
        new ceres::NumericDiffCostFunction<rlxTrackingError, ceres::CENTRAL, 1, 4>
        (new rlxTrackingError(string(argv[1])));
    
    problem.AddResidualBlock(cost_function,
                             NULL,
                             params);

    ceres::Solver::Options options;
    options.minimizer_type = ceres::LINE_SEARCH;
    options.line_search_direction_type = ceres::BFGS;
    options.minimizer_progress_to_stdout = true;
    //options.max_num_iterations = pinhole_max_iterations;
    
    options.min_line_search_step_size = 0.1;
    //options.numeric_derivative_relative_step_size = 0.5;

    int threads = omp_get_num_procs();
    options.num_threads = threads;
    //cout<<"\nSolver using "<<threads<<" threads.\n\n";

    //options.gradient_tolerance = 1E-15;
    //options.function_tolerance = 1E-8;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout<<summary.FullReport()<<"\n";
    
    return 1;
    */
}
