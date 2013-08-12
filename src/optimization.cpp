// -------------------------------------------------------
// -------------------------------------------------------
// Synthetic Aperture - Particle Tracking Velocimetry Code
// --- Optimization Function Library ---
// -------------------------------------------------------
// Author: Abhishek Bajpayee
//         Dept. of Mechanical Engineering
//         Massachusetts Institute of Technology
// -------------------------------------------------------
// -------------------------------------------------------

#include "std_include.h"
#include "optimization.h"

using namespace cv;
using namespace std;

// Pinhole bundle adjustment function
double BA_pinhole(baProblem &ba_problem, string ba_file, Size img_size, vector<int> const_points) {

    cout<<"\nRUNNING PINHOLE BUNDLE ADJUSTMENT TO CALIBRATE CAMERAS...\n";
    //google::InitGoogleLogging(argv);
    
    if (!ba_problem.LoadFile(ba_file.c_str())) {
        std::cerr << "ERROR: unable to open file " << ba_file << "\n";
        return 1;
    }

    ba_problem.cx = img_size.width*0.5;
    ba_problem.cy = img_size.height*0.5;
    
    // Create residuals for each observation in the bundle adjustment problem. The
    // parameters for cameras and points are added automatically.
    ceres::Problem problem;
    for (int i = 0; i < ba_problem.num_observations(); ++i) {
        // Each Residual block takes a point and a camera as input and outputs a 2
        // dimensional residual. Internally, the cost function stores the observed
        // image location and compares the reprojection against the observation.
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<pinholeReprojectionError, 3, 9, 3, 4>
            (new pinholeReprojectionError(ba_problem.observations()[2 * i + 0],
                                          ba_problem.observations()[2 * i + 1],
                                          ba_problem.cx, ba_problem.cy, ba_problem.num_cameras()));
        
        problem.AddResidualBlock(cost_function,
                                 NULL,
                                 ba_problem.mutable_camera_for_observation(i),
                                 ba_problem.mutable_point_for_observation(i),
                                 ba_problem.mutable_plane_for_observation(i));

        /*
        for (int j=0; j<const_points.size(); j++) {
            if (ba_problem.point_index()[i]==const_points[j]) {
                cout<<"Setting point "<<ba_problem.point_index()[i]<<" constant"<<endl;
                problem.SetParameterBlockConstant(ba_problem.mutable_point_for_observation(i));
            }
        }
        */

    }
    
    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 50;
    
    int threads = omp_get_num_procs();
    options.num_threads = threads;
    cout<<"\nSolver using "<<threads<<" threads.\n\n";

    options.gradient_tolerance = 1E-12;
    options.function_tolerance = 1E-8;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout<<summary.FullReport()<<"\n";
    cout<<"BUNDLE ADJUSTMENT COMPLETE!\n\n";

    return double(summary.final_cost);

}

// Refractivee bundle adjustment function
double BA_refractive(baProblem_ref &ba_problem, string ba_file, Size img_size, vector<int> const_points) {

    cout<<"\nRUNNING REFRACTIVE BUNDLE ADJUSTMENT TO CALIBRATE CAMERAS...\n";
    //google::InitGoogleLogging(argv);
    
    if (!ba_problem.LoadFile(ba_file.c_str())) {
        std::cerr << "ERROR: unable to open file " << ba_file << "\n";
        return 1;
    }

    ba_problem.cx = img_size.width*0.5;
    ba_problem.cy = img_size.height*0.5;
    
    // Create residuals for each observation in the bundle adjustment problem. The
    // parameters for cameras and points are added automatically.
    ceres::Problem problem;
    for (int i = 0; i < ba_problem.num_observations(); ++i) {
    //for (int i = 0; i < 2; ++i) {
        // Each Residual block takes a point and a camera as input and outputs a 2
        // dimensional residual. Internally, the cost function stores the observed
        // image location and compares the reprojection against the observation.
        ceres::CostFunction* cost_function =
            new ceres::NumericDiffCostFunction<refractiveReprojectionError, ceres::CENTRAL, 3, 9, 3, 4>
            (new refractiveReprojectionError(ba_problem.observations()[2 * i + 0],
                                             ba_problem.observations()[2 * i + 1],
                                             ba_problem.cx, ba_problem.cy, 
                                             ba_problem.num_cameras(),
                                             ba_problem.t(), ba_problem.n1(), ba_problem.n2(), ba_problem.n3() ));
        
        problem.AddResidualBlock(cost_function,
                                 NULL,
                                 ba_problem.mutable_camera_for_observation(i),
                                 ba_problem.mutable_point_for_observation(i),
                                 ba_problem.mutable_plane_for_observation(i));
        
        /*
        for (int j=0; j<const_points.size(); j++) {
            if (ba_problem.point_index()[i]==const_points[j]) {
                cout<<"Setting point "<<ba_problem.point_index()[i]<<" constant"<<endl;
                problem.SetParameterBlockConstant(ba_problem.mutable_point_for_observation(i));
            }
        }
        */

    }
    
    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    
    int threads = omp_get_num_procs();
    options.num_threads = threads;
    cout<<"\nSolver using "<<threads<<" threads.\n\n";

    options.gradient_tolerance = 1E-12;
    options.function_tolerance = 1E-8;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout<<summary.FullReport()<<"\n";
    cout<<"BUNDLE ADJUSTMENT COMPLETE!\n\n";

    return double(summary.final_cost);

}

double fit_plane(leastSquares &ls_problem, string filename, vector<Mat> points) {

    cout<<"\nFITTING PLANE TO CAMERA LOCATIONS...\n";

    if (!ls_problem.LoadFile(filename.c_str())) {
        std::cerr << "ERROR: unable to open file " << filename << "\n";
        return 1;
    }

    // Create residuals for each observation in the bundle adjustment problem. The
    // parameters for cameras and points are added automatically.
    ceres::Problem problem;
    for (int i=0; i < points.size(); ++i) {
        
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<planeError, 1, 4>
            (new planeError(points[i].at<double>(0,0),
                            points[i].at<double>(0,1),
                            points[i].at<double>(0,2)));
        
        problem.AddResidualBlock(cost_function,
                                 NULL,
                                 ls_problem.mutable_params());

    }
    
    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 50;
    options.num_threads = 16;
    options.gradient_tolerance = 1E-12;
    options.function_tolerance = 1E-8;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout<<summary.FullReport()<<"\n";
    cout<<"ROTATION MATRIX CALCULATED!\n";
    return summary.final_cost;

}
