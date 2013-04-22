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
double BA_pinhole(baProblem &ba_problem, string ba_file, Size img_size) {

    cout<<"\nRUNNING BUNDLE ADJUSTMENT TO CALIBRATE CAMERAS...\n";
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
    }
    
    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 50;
    options.num_threads = 16;
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

// Bundle adjustment result alignment function
double BA_align(alignProblem &align_problem, string align_file) {

    cout<<"\nALIGNING BUNDLE ADJUSTMENT RESULT...\n";
    
    if (!align_problem.LoadFile(align_file.c_str())) {
        std::cerr << "ERROR: unable to open file " << align_file << "\n";
        return 1;
    }
    
    align_problem.initialize();
    
    // Create residuals for each camera.
    ceres::Problem problem;
    const int num_params = 6*align_problem.num_cameras();

    // The residual block
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<alignmentError, 27, 54>
        (new alignmentError(align_problem.constants_r(), align_problem.num_cameras()));
    
    problem.AddResidualBlock(cost_function,
                             NULL,
                             align_problem.mutable_params());
    
    // Solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 50;
    options.num_threads = 16;
    options.gradient_tolerance = 1E-12;
    options.function_tolerance = 1E-8;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout<<summary.FullReport()<<"\n";
    cout<<"ALIGNMENT COMPLETE!\n\n";

    return double(summary.final_cost);
    
}

/*
// Refractive bundle adjustment function
double BA_pinhole(baProblem &ba_problem, string ba_file, char* argv) {

    cout<<"\nRUNNING BUNDLE ADJUSTMENT TO REFINE CALIBRATION...\n";
    google::InitGoogleLogging(argv);
    
    if (!ba_problem.LoadFile(ba_file.c_str())) {
        std::cerr << "ERROR: unable to open file " << ba_file << "\n";
        return 1;
    }
    
    // Create residuals for each observation in the bundle adjustment problem. The
    // parameters for cameras and points are added automatically.
    ceres::Problem problem;
    for (int i = 0; i < ba_problem.num_observations(); ++i) {
        // Each Residual block takes a point and a camera as input and outputs a 2
        // dimensional residual. Internally, the cost function stores the observed
        // image location and compares the reprojection against the observation.
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<pinholeReprojectionError, 1, 9, 3, 5>(
                                                                               new pinholeReprojectionError(
                                                                                                            ba_problem.observations()[2 * i + 0],
                                                                                                            ba_problem.observations()[2 * i + 1]));
        
        problem.AddResidualBlock(cost_function,
                                 NULL,
                                 ba_problem.mutable_camera_for_observation(i),
                                 ba_problem.mutable_point_for_observation(i)
                                 ba_problem.mutable_scene_params());
    }
    
    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 50;
    options.num_threads = 16;
    options.gradient_tolerance = 1E-12;
    options.function_tolerance = 1E-8;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout<<summary.FullReport()<<"\n";
    cout<<"BUNDLE ADJUSTMENT COMPLETE!\n";

    return double(summary.final_cost);

}
*/
