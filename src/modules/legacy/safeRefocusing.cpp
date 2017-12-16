// Written by Barry Scharfman

#include "safeRefocusing.h"

// #include "pcl3DExtracting.h"

// #include <pcl/ModelCoefficients.h>
// #include <pcl/point_types.h>
// #include <pcl/io/pcd_io.h>
// #include <pcl/filters/extract_indices.h>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/kdtree/kdtree.h>
// #include <pcl/sample_consensus/method_types.h>
// #include <pcl/sample_consensus/model_types.h>
// #include <pcl/segmentation/sac_segmentation.h>

using namespace std;

safeRefocus::safeRefocus(refocus_settings settings, safe_refocus_settings safe_settings):saRefocus(settings), dp(safe_settings.dp), minDist(safe_settings.minDist), param1(safe_settings.param1), param2(safe_settings.param2), minRadius(safe_settings.minRadius), maxRadius(safe_settings.maxRadius), gKerHt(safe_settings.gKerHt), gKerWid(safe_settings.gKerWid), gKerSigX(safe_settings.gKerSigX), gKerSigY(safe_settings.gKerSigY), circle_rim_thickness(safe_settings.circle_rim_thickness), debug(safe_settings.debug) {}

void safeRefocus::safe_preprocess_all(){
  // Print circle detection input parameters
  LOG(INFO)<<"Circle detection parameters:"<<endl;
  LOG(INFO)<<"dp: "<<dp<<endl;
  LOG(INFO)<<"minDist: "<<minDist<<endl;
  LOG(INFO)<<"param1: "<<param1<<endl;
  LOG(INFO)<<"param2: "<<param2<<endl;
  LOG(INFO)<<"minRadius: "<<minRadius<<endl;
  LOG(INFO)<<"maxRadius: "<<maxRadius<<endl;

  // Print Gaussian blur input parameters
  LOG(INFO)<<"Gaussian blur parameters:"<<endl;
  LOG(INFO)<<"gKerWid: "<<gKerWid<<endl;
  LOG(INFO)<<"gKerHt: "<<gKerHt<<endl;
  LOG(INFO)<<"gKerSigX: "<<gKerSigX<<endl;
  LOG(INFO)<<"gKerSigY: "<<gKerSigY<<endl;

  LOG(INFO)<<"Number of raw data images: "<<imgs.size()<<endl;

  for (int j=0; j<imgs.size(); j++) {
    for (int k=0; k<imgs[j].size(); k++){
				
      Mat raw_image = imgs[j][k]; // imgs is a variable from the saRefocus class. First index is cam #, second index is frame #
      Mat processed_image; 
      safe_preprocess_single_img(raw_image, processed_image);
      //processed_image.convertTo(processed_image, CV_8U);
      imgs[j][k] = processed_image.clone();

      LOG(INFO)<<"imgs["<<j<<"]["<<k<<"] type: "<<imgs[j][k].type()<<endl;
    }	
		
  }
}

void safeRefocus::safe_preprocess_single_img(Mat in, Mat &out) {

  LOG(INFO)<<"Running preprocess_single_img function in safeRefocusing!!**************"<<endl;
    
  //debug_imshow(in, "Raw input image");
	
  /// Convert in image to grayscale if it is not already
  Mat in_CV_8U = in.clone();  
  in_CV_8U.convertTo(in_CV_8U, CV_8U);

  //debug_imshow(in_CV_8U, "in_CV_8U");

  /// Use Gaussian filter to reduce noise to avoid false circle detection  
  if(gKerWid > 0 && gKerHt > 0){
    LOG(INFO)<<"Performing Gaussian blur:"<<endl;
    cv::GaussianBlur( in_CV_8U, in_CV_8U, Size(gKerWid, gKerHt), gKerSigX, gKerSigY ); // (3,3), 2 2
  }

  // Detect circles
  featureDetect fD_for_find_circles(in_CV_8U);
  vector<double> x_ctr, y_ctr, rad;
  int num_circles_found;
  fD_for_find_circles.find_circles_in_img(x_ctr, y_ctr, rad, num_circles_found, dp, minDist, param1, param2, minRadius, maxRadius);	
    
  /// Draw the circles detected 
  int rows = in.rows;
  int cols = in.cols;
  out = Mat::zeros(rows, cols, CV_8U);
  featureDetect fD_for_draw_circles(out, out);
  fD_for_draw_circles.draw_circles_on_img(num_circles_found, x_ctr, y_ctr, rad, circle_rim_thickness);
   
  // out.convertTo(out, CV_8U);
  // Show binary image with detected circles:
  double min, max;
  minMaxLoc(out, &min, &max);
  LOG(INFO)<<"Min is: "<<min<<endl;
  LOG(INFO)<<"Max is: "<<max<<endl;  

  debug_imshow(out, "Circles overlay image");

  // Get nonzero coordinates:
  // out.convertTo(out, CV_8UC1);
  // featureDetect fD_getNonZeroCoords(out);
  // vector<double> x_nonzero, y_nonzero;
  // fD_getNonZeroCoords.getNonZeroCoords(x_nonzero, y_nonzero);

  // LOG(INFO)<<"Nonzero coordinates in preprocessed image:"<<endl;
  // for (int k=0; k<x_nonzero.size(); k++){
  //   LOG(INFO)<<x_nonzero[k]<<"\t"<<y_nonzero[k]<<endl;
  // }
  
}

void safeRefocus::perform_SAFE(string save_path, double zmin, double zmax, double dz, double thresh) {
  int num_frames = imgs[0].size();

  LOG(INFO)<<"Number of frames to refocus and process: "<<num_frames<<endl;
    
  // Create output save file 
  stringstream s;
  s<<save_path<<"output.txt";
  ofstream file;
  file.open(s.str().c_str());  

  //Write a top line in file
  file<<"First line"<<endl;

  // Define overall vectors of vectors for circle centroids and radii over all frames
  // vector< vector<double> > x_ctr_overall, y_ctr_overall, z_ctr_overall, rad_overall;

  // For imgs, first index is cam #, second index is frame #
  //debug_imshow(imgs[0][0], "Input imgs[0][0] JUST BEFORE REFOCUS");

  // Loop over all frames  
  for (int f=0; f<num_frames; f++) { 
        
    // // Create and set up a cloud object to hold all 
    // pcl::PointCloud<pcl::PointXYZ> currFrameCloud;
    // currFrameCloud.height = 1;
    // currFrameCloud.is_dense = true;
    // currFrameCloud.points.resize (currFrameCloud.width * currFrameCloud.height);
          
    // Loop over all requested depths and build cloud of nonzero voxels
    
    // Set up vectors for current frame found circles centroids and radii:
    // vector<double> x_ctr_curr_frame, y_ctr_curr_frame, z_ctr_curr_frame, rad_curr_frame;

    LOG(INFO)<<"Refocusing frame #"<<f<<"...";
    for (double z=zmin; z<=zmax; z+=dz) {
      LOG(INFO)<<"z = "<<z<<"..." <<endl;

      double min, max;
      minMaxLoc(imgs[0][0], &min, &max);
      LOG(INFO)<<"image used to refocus Min is: "<<min<<endl;
      LOG(INFO)<<"image used to refocus Max is: "<<max<<endl;
      LOG(INFO)<<"image used to refocus is of type: "<<imgs[0][0].type()<<endl;    
      
      // Refocus current depth of current frame - result stored in result Mat variable from saRefocus parent class	
      Mat result = refocus(z, 0, 0, 0, thresh, f); // TODO: update the (0,0,0) for desired refocus angles in saRefocus class

      //debug_imshow(result, "Raw result image");
      
      // double min, max;
      minMaxLoc(result, &min, &max);
      LOG(INFO)<<"refocused result Min is: "<<min<<endl;
      LOG(INFO)<<"refocused result Max is: "<<max<<endl;

      Mat result_raw = result.clone(); // store raw refocused image at current depth

      // Convert result array to 8-bit array for HoughCircles function
      Mat result_gray; // = Mat::zeros(3,3, CV_8UC1);
      result.convertTo(result_gray, CV_8UC1, 255.0);
      LOG(INFO)<<"result Mat is of type: "<<result.type()<<endl;
      LOG(INFO)<<"result_gray Mat is of type: "<<result_gray.type()<<endl;

      // Process refocused image at current depth:
      // ****maybe add morphological processing here
      // int kernel_wid_morph = 3;
      // int kernel_wid_morph = 3;
      Mat kernel_morph = Mat();
      int iterations_morph=1;
      int borderType_morph=BORDER_CONSTANT;

      featureDetect fD_getNonZeroCoords(result_gray);
      vector<double> x_nonzero, y_nonzero;
      fD_getNonZeroCoords.getNonZeroCoords(x_nonzero, y_nonzero);
      if(!(x_nonzero.empty() && y_nonzero.empty())){ 
	LOG(INFO)<<"z = "<<z<<" has nonzero coordinates."<<endl;
	LOG(INFO)<<"Nonzero coordinates in refocused image:"<<endl;
	for (int k=0; k<x_nonzero.size(); k++){
	  // LOG(INFO)<<x_nonzero[k]<<"\t"<<y_nonzero[k]<<endl;
	}
	dilate(result, result, kernel_morph, Point(-1,-1), iterations_morph, 2, 1);
	debug_imshow(result, "Result dilated nonzero image");

      }
            
      erode(result, result, kernel_morph, Point(-1,-1), iterations_morph, 2, 1);
      
      // Use Gaussian filter to reduce noise to avoid false circle detection for refocused image at current depth: 
      if(gKerWid > 0 && gKerHt > 0){
	GaussianBlur( result_gray, result_gray, Size(gKerWid, gKerHt), gKerSigX, gKerSigY );
      }      
      
      // Detect circles in refocused image at current depth:
      featureDetect fD_for_find_circles(result_gray);
      vector<double> x_ctr, y_ctr, rad;
      int num_circles_found;
      fD_for_find_circles.find_circles_in_img(x_ctr, y_ctr, rad, num_circles_found, dp, minDist, param1, param2, minRadius, maxRadius);	
      
      // Write circle centroids and radii for current frame to file
      if(num_circles_found>0){
	for (int k=0;k<num_circles_found;k++){
	  file<<x_ctr[k]<<"\t";
	  file<<y_ctr[k]<<"\t";
	  file<<z<<"\t";
	  file<<rad[k]<<"\t";
	  file<<f<<endl;
	}
      }

      // Add circle centroids and radii to vectors for current frame
      // x_ctr_curr_frame.push_back(x_ctr);
      // y_ctr_curr_frame.push_back(y_ctr);
      // z_ctr_curr_frame.push_back(z);
      // rad_curr_frame.push_back(rad);
      
      // // Extract nonzero pixel coordinates and store in all_x_curr_img and all_y_curr_img, respectively
      // vector<double> all_x_curr_img;
      // vector<double> all_y_curr_img;
      // getNonZeroCoords(result, all_x_curr_img, all_y_curr_img);
    	       	    
      // // Add nonzero pixels from current image to cloud
      // for (int j=0; j<all_x_curr_img.size(); j++) {
      // 	pcl::PointXYZ point;
      // 	point.x = all_x_curr_img[j];
      // 	point.y = all_y_curr_img[j];
      // 	point.z = z; // The z coordinate is that of the current refocused depth
      // 	currFrameCloud.points.push_back(point);
      // }
        
      // currFrameCloud.width = (uint32_t) currFrameCloud.points.size();
        
      // // Find 3D blobs in cloud for current frame
      // // pcl3DExtract pcl3D;
      // //
      // pcl3D.getClusters(currFrameCloud, tol, minSize, maxSize, output_path); // currFrameCloud
            
      // Get centroids of 3D blobs for current frame
            
            
      // Get radii of 3D blobs for current frame
            

      LOG(INFO)<<"Done processing frame "<<f<<endl;

    }

    // Add circle centroids and radii from current frame to overall vectors for over time
    // x_ctr_overall.push_back(x_ctr_curr_frame);
    // y_ctr_overall.push_back(y_ctr_curr_frame); 
    // z_ctr_overall.push_back(z_ctr_curr_frame);
    // rad_overall.push_back(rad_curr_frame);

    LOG(INFO)<<"Done processing all frames!"<<endl;

  }

  // Close output save file
  file.close();
}

void safeRefocus::debug_imshow(Mat img, string title) {
  // If debug mode is on, show input img with title

  if(debug){
    imshow( title, img );
    waitKey(0);
  }
}








