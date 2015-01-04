// Written by Barry Scharfman

#include "std_include.h"
#include "tools.h"
#include "cuda_lib.h"
#include "visualize.h"

#include <Eigen/Core>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

#include <stdlib.h>

#include "featureDetection.h"

using namespace std;
using namespace cv;

featureDetect::featureDetect(Mat inMat_, Mat &outMat_):inMat(inMat_),outMat(outMat_){}

featureDetect::featureDetect(Mat inMat_):inMat(inMat_){}

void featureDetect::detect_edges_canny(){
  LOG(INFO)<<"Performing Canny edge detection..."<<endl;

  // initialize the output matrix with zeros
  Mat new_image = Mat::zeros( inMat.size(), inMat.type() );
 
  // create a matrix with all elements equal to 255 for subtraction
  Mat sub_mat = Mat::ones(inMat.size(), inMat.type())*255;
 
  //subtract the original matrix by sub_mat to give the negative output new_image
  subtract(sub_mat, inMat, new_image);

  Mat dst, detected_edges;

  int edgeThresh = 1;
  int lowThreshold = 40;
  int ratio = 2;
  int kernel_size = 3;

  /// Reduce noise with a kernel 3x3
  blur( inMat, detected_edges, Size(3,3) );

  imshow( "Inverted Image", detected_edges );
  waitKey(0);

  /// Canny detector
  Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  // Using Canny's output as a mask, we display our result
  dst = Scalar::all(0);
  inMat.copyTo( dst, detected_edges);

  outMat = detected_edges;
  imshow( "Edge Map", outMat );
  waitKey(0); 
}

void featureDetect::find_circles_in_img(vector<double> &x_ctr, vector<double> &y_ctr, vector<double> &rad, int &num_circles_found, double dp, double minDist, double param1, double param2, int minRadius, int maxRadius){

  //circles: A vector that stores sets of 3 values: x_{c}, y_{c}, r for each detected circle.
  //CV_HOUGH_GRADIENT: Define the detection method. Currently this is the only one available in OpenCV
  //dp: The inverse ratio of resolution
  //min_dist: Minimum distance between detected centers
  //param_1: Upper threshold for the internal Canny edge detector
  //param_2: Threshold for center detection.
  //min_radius: Minimum radio to be detected. If unknown, put zero as default.
  //max_radius: Maximum radius to be detected. If unknown, put zero as default

  //

  vector<Vec3f> circles;

  // Show current image
  // imshow( "Image in which circles will be detected", inMat );
  // waitKey(0);

  // Detect Circles
  HoughCircles( inMat, circles, CV_HOUGH_GRADIENT, dp, minDist, param1, param2, minRadius, maxRadius );

  // Count and report number of circles detected 
  num_circles_found = circles.size();

  if (num_circles_found == 0) {
    LOG(INFO)<<"No circles detected."<<endl;
    return;
  }else{
    LOG(INFO)<<num_circles_found<<" circle(s) detected."<<endl;
  }

  // Add circle detection info to centroid and radii vectors
  for( int i = 0; i < num_circles_found; i++ ){
    x_ctr.push_back(circles[i][0]);
    y_ctr.push_back(circles[i][1]);
    rad.push_back(circles[i][2]);
  }

}

void featureDetect::draw_circles_on_img(int num_circles_found, vector<double> x_ctr, vector<double> y_ctr, vector<double> rad, int circle_rim_thickness){
  
  /// Draw the circles detected 
  for( int i = 0; i < num_circles_found; i++ )
    {
      Point center(cvRound(x_ctr[i]), cvRound(y_ctr[i]));
      int radius = cvRound(rad[i]);
      LOG(INFO)<<"Center (x,y) = ("<<x_ctr[i]<<", "<<y_ctr[i]<<")"<<endl;
      LOG(INFO)<<"Radius = "<<rad[i]<<endl;

      // Draw Circle Center
      //cv::circle( outMat, center, 3, Scalar(0,255,0), -1, 8, 0 );
      
      // Draw Circle Outline
      cv::circle( outMat, center, radius, Scalar(255,255,255), circle_rim_thickness, 8, 0 ); // 0
    }
}

void featureDetect::find_MSERs_in_img(int _delta, int _min_area, int _max_area,
				    float _max_variation, float _min_diversity,
				    int _max_evolution, double _area_threshold,
				    double _min_margin, int _edge_blur_size){

  // MSER parameters
  // Delta delta, in the code, it compares (size_{i}-size_{i-delta})/size_{i-delta}. default 5. 
  // MinArea prune the area which smaller than minArea. default 60.
  // MaxArea prune the area which bigger than maxArea. default 14400.
  // MaxVariation prune the area have simliar size to its children. default 0.25
  // MinDiversity trace back to cut off mser with diversity < min_diversity. default 0.2.
  // MaxEvolution for color image, the evolution steps. default 200.
  // AreaThreshold the area threshold to cause re-initialize. default 1.01.
  // MinMargin ignore too small margin. default 0.003.
  // EdgeBlurSize the aperture size for edge blur. default 5.
  // Mask Optional input mask that marks the regions where we should detect features

  // int _delta = 5;
  // int _min_area = 100;
  // int _max_area = 3000; // 14400
  // float _max_variation = 0.4; // 0.25
  // float _min_diversity = 0.5; // 0.2
  // int _max_evolution = 200;
  // double _area_threshold = 2.01; // 1.01;
  // double _min_margin = 0.001; // 0.003
  // int _edge_blur_size = 1; // 5

  static const Vec3b bcolors[] =
    {
      Vec3b(0,0,255),
      Vec3b(0,128,255),
      Vec3b(0,255,255),
      Vec3b(0,255,0),
      Vec3b(255,128,0),
      Vec3b(255,255,0),
      Vec3b(255,0,0),
      Vec3b(255,0,255),
      Vec3b(255,255,255)
    };

  vector<vector<cv::Point> > contours;
  Mat ellipses;
  inMat.copyTo(ellipses);

  MSER(_delta, _min_area, _max_area,
	   _max_variation, _min_diversity,
	   _max_evolution, _area_threshold,
	   _min_margin, _edge_blur_size)(ellipses, contours);

    // draw mser's with different colors
    for( int i = (int)contours.size()-1; i >= 0; i-- )
    {
      const vector<Point>& r = contours[i];
        for ( int j = 0; j < (int)r.size(); j++ )
        {
            Point pt = r[j];
            ellipses.at<Vec3b>(pt) = bcolors[i%9];
        }

        // find ellipse (it seems cvfitellipse2 have error or sth?)
        RotatedRect box = fitEllipse( r );
	Size2f rot_rect_size = box.size;
	cout<<"Rotated rectange sizes:"<<endl;
	cout<<rot_rect_size<<endl;
	//cout<<rot_rect_size(2)<<endl;

        box.angle=(float)CV_PI/2-box.angle;
        ellipse( ellipses, box, Scalar(196,255,255), 2 );
    }

    imshow( "ellipses", ellipses );

    waitKey(0);  

}

void featureDetect::getNonZeroCoords(vector<double> &all_x, vector<double> &all_y){
  Mat nonZeroCoordinates;
  cv::findNonZero(inMat, nonZeroCoordinates);
	
  for (int i = 0; i < nonZeroCoordinates.total(); i++ ) {
    // add x and y coordinates of current point to overall all_x and all_y vectors
    all_x.push_back(nonZeroCoordinates.at<Point>(i).x);
    all_y.push_back(nonZeroCoordinates.at<Point>(i).y);        	
  }	
}
