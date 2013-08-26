// -------------------------------------------------------
// -------------------------------------------------------
// Synthetic Aperture - Particle Tracking Velocimetry Code
// --- Calibration Library ---
// -------------------------------------------------------
// Author: Abhishek Bajpayee
//         Dept. of Mechanical Engineering
//         Massachusetts Institute of Technology
// -------------------------------------------------------
// -------------------------------------------------------

#ifndef CALIBRATION_LIBRARY
#define CALIBRATION_LIBRARY

#include "std_include.h"
#include "optimization.h"
#include "typedefs.h"

class camCalibration {

 public:
    
    ~camCalibration() {
        //
    }

    camCalibration() {
        //
    }

    // Function declarations
    void calibrateCamera(vector< vector<Point2f> > corner_points, Size img_size, Mat &K, Mat &rvec, Mat &tvec);

 protected:

    // Inheritable stuff

 private: 

    // Private functions and variables

};

class multiCamCalibration {
 
 public:
    ~multiCamCalibration() {
        //
    }

 multiCamCalibration(string path, Size grid_size, double grid_size_phys, int refractive, int dummy_mode, int dir_struct):
    path_(path), grid_size_(grid_size), grid_size_phys_(grid_size_phys), dummy_mode_(dummy_mode), refractive_(refractive), dir_struct_(dir_struct) {}
    
    // Functions to access calibration data
    int num_cams() { return num_cams_; }
    int num_imgs() { return num_imgs_; }
    Size grid_size() { return grid_size_; }
    vector<string> cam_names() { return cam_names_; }
    refocusing_data refocusing_params() { return refocusing_params_; }
    
    // Functions to run calibration
    void initialize();
    void run();
    
    // Functions to work with multipage tiff files
    void read_cam_names_mtiff();
    void read_calib_imgs_mtiff();

    // Functions to work with normal directory structure
    void read_cam_names();
    void read_calib_imgs();

    void find_corners();
    void initialize_cams();
    
    void initialize_cams_ref();

    void write_BA_data();
    void write_BA_data_ref();
    void run_BA();
    void run_BA_ref();

    double run_BA_pinhole(baProblem &ba_problem, string ba_file, Size img_size, vector<int> const_points);
    double run_BA_refractive(baProblem_ref &ba_problem, string ba_file, Size img_size, vector<int> const_points);

    void write_calib_results();
    void write_calib_results_ref();
    void load_calib_results();

    void write_calib_results_matlab();
    void write_calib_results_matlab_ref();

    void grid_view();

 private:

    void calc_space_warp_factor();
    void get_grid_size_pix();

    string path_;
    string ba_file_;
    string result_dir_;
    string result_file_;
    
    int num_cams_;
    int num_imgs_;
    int center_cam_id_;
    int origin_image_id_;
    
    Size grid_size_;
    Size img_size_;
    
    double grid_size_phys_;
    double grid_size_pix_;
    double pix_per_phys_;
    double warp_factor_;
    
    vector<string> cam_names_;
    vector< vector<Mat> > calib_imgs_;
    vector< vector< vector<Point3f> > > all_pattern_points_;
    vector< vector< vector<Point2f> > > all_corner_points_;
    vector<Mat> cameraMats_;
    vector<Mat> dist_coeffs_;
    vector< vector<Mat> > rvecs_;
    vector< vector<Mat> > tvecs_;
    
    vector<Mat> rVecs_;
    vector<Mat> tVecs_;
    vector<Mat> K_mats_;
    vector<Mat> dist_mats_;
    
    vector<int> const_points_;

    refocusing_data refocusing_params_;

    baProblem ba_problem_;
    baProblem_ref ba_problem_ref_;
    double total_reproj_error_;
    double total_error_;
    double avg_reproj_error_;
    double avg_error_;

    // Option flags
    int solveForDistortion; // TODO: NOT IMPLEMENTED
    int squareGrid; // TODO: NOT IMPLEMENTED
    int saveCornerImgs; // TODO: NOT IMPLEMENTED
    int show_corners_flag;
    int run_calib_flag;
    int results_just_saved_flag;
    int load_results_flag;
    int dummy_mode_;
    int refractive_;
    int dir_struct_; // 0 = folders with tif images, 1 = multipage tif files

    int pinhole_max_iterations;
    int refractive_max_iterations;
    
};
    
#endif
