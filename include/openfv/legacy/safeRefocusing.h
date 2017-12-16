// Written by Barry Scharfman, December 2014

#ifndef SAFE_REFOCUSING_LIBRARY
#define SAFE_REFOCUSING_LIBRARY

#include "std_include.h"
#include "calibration.h"
#include "typedefs.h"
#include "tools.h"
#include "visualize.h"
#include "refocusing.h"

#ifndef WITHOUT_CUDA
#include "cuda_lib.h"
#endif

#include <Eigen/Core>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <unistd.h>

#include "featureDetection.h"

/* #include "pcl3DExtracting.h" */

/* #include <pcl/ModelCoefficients.h> */
/* #include <pcl/point_types.h> */
/* #include <pcl/io/pcd_io.h> */
/* #include <pcl/filters/extract_indices.h> */
/* #include <pcl/filters/voxel_grid.h> */

/* #include <pcl/kdtree/kdtree.h> */
/* #include <pcl/sample_consensus/method_types.h> */
/* #include <pcl/sample_consensus/model_types.h> */
/* #include <pcl/segmentation/sac_segmentation.h> */


class safeRefocus : public saRefocus {

 public:
    ~safeRefocus() {

    }
    
    safeRefocus(refocus_settings settings, safe_refocus_settings safe_settings);

    void safe_preprocess_all();
       
    void safe_preprocess_single_img(Mat in, Mat &out);

    void find_circles_in_img(vector<double> &x_ctr, vector<double> &y_ctr, vector<double> &rad, int &num_circles_found, double dp, double minDist, double param1, double param2, int minRadius, int maxRadius);
    
    void perform_SAFE(string save_path, double zmin, double zmax, double dz, double thresh);
    
    void getNonZeroCoords(Mat inMat, vector<double> &all_x, vector<double> &all_y);
    
    void find_MSERs_in_img(Mat inMat, int _delta, int _min_area, int _max_area,
				    float _max_variation, float _min_diversity,
				    int _max_evolution, double _area_threshold,
			   double _min_margin, int _edge_blur_size);

    void debug_imshow(Mat img, string title);

 private:

    double dp;
    double minDist;
    double param1;
    double param2;
    int minRadius;
    int maxRadius;
    int gKerWid;
    int gKerHt;
    int gKerSigX;
    int gKerSigY;
    int circle_rim_thickness;
    int debug;

};

#endif
