// Written by Barry Scharfman, December 2014

#ifndef FEATURE_DETECTION_LIBRARY
#define FEATURE_DETECTION_LIBRARY

#include "std_include.h"
#include "tools.h"
#include "visualize.h"

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

#include <stdlib.h>

class featureDetect {

 public:
    ~featureDetect() {

    }
    
    featureDetect(Mat inMat_, Mat &outMat_);

    featureDetect(Mat inMat);
       
    void detect_edges_canny();

    void find_circles_in_img(vector<double> &x_ctr, vector<double> &y_ctr, vector<double> &rad, int &num_circles_found, double dp, double minDist, double param1, double param2, int minRadius, int maxRadius);
        
    void draw_circles_on_img(int num_circles_found, vector<double> x_ctr, vector<double> y_ctr, vector<double> rad, int circle_rim_thickness);

    void getNonZeroCoords(vector<double> &all_x, vector<double> &all_y);
    
    void find_MSERs_in_img(int _delta, int _min_area, int _max_area,
			   float _max_variation, float _min_diversity,
			   int _max_evolution, double _area_threshold,
			   double _min_margin, int _edge_blur_size);

 private:
    Mat inMat;
    Mat outMat;

    double dp;
    double minDist;
    double param1;
    double param2;
    int minRadius;
    int maxRadius;

};

#endif
