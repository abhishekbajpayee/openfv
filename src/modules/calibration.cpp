//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//                           License Agreement
//                For Open Source Flow Visualization Library
//
// Copyright 2013-2015 Abhishek Bajpayee
//
// This file is part of openFV.
//
// openFV is free software: you can redistribute it and/or modify it under the terms of the 
// GNU General Public License as published by the Free Software Foundation, either version 
// 3 of the License, or (at your option) any later version.
//
// openFV is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
// See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with openFV. 
// If not, see http://www.gnu.org/licenses/.

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

#include "calibration.h"
#include "optimization.h"
#include "tools.h"

using namespace libtiff;

multiCamCalibration::multiCamCalibration(string path, Size grid_size, double grid_size_phys, int refractive, int dummy_mode, int mtiff, int mp4, int skip, int start_frame, int end_frame, int show_corners): path_(path), grid_size_(grid_size), grid_size_phys_(grid_size_phys), dummy_mode_(dummy_mode), refractive_(refractive), mtiff_(mtiff), mp4_(mp4), skip_frames_(skip), start_frame_(start_frame), end_frame_(end_frame), show_corners_flag(show_corners) {

    // Standard directories and filenames
    // TODO: this use of file needs to go
    ba_file_ = string("tmp_ba_data.txt");
    result_dir_ = string("calibration_results");

    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL)
        LOG(INFO)<<"Current working dir: "<<string(cwd);
    else
        LOG(FATAL)<<"getcwd() error!";
    
    run_calib_flag = 0;
    load_results_flag = 0;

    int result_dir_found = 0;
    DIR *dir;
    struct dirent *ent;
    string temp_name;
    dir = opendir(path_.c_str());

    int choice;

    while(ent = readdir(dir)) {
        temp_name = ent->d_name;
        if (!temp_name.compare(result_dir_)) {
            result_dir_found = 1;
            LOG(INFO)<<"\'"<<result_dir_<<"\' directory found in "<<path_<<"! Calibration has already been performed earlier. Please select what to do...\n1) Run calibration again\n2) Read results\nEnter your choice (1/2): ";
            cin>>choice;
            if (choice==1) {
                run_calib_flag = 1;
            } else if (choice==2) {
                load_results_flag = 1;
            } else {
                LOG(INFO)<<"Invalid choice!\n";
            }
        }
    }
    if (!result_dir_found) run_calib_flag = 1;

    // show_corners_flag = 0;
    results_just_saved_flag = 0;

    // Default settings for optimization routines
    pinhole_max_iterations = 100;
    refractive_max_iterations = 100;
    init_f_value_ = 2500;

    // Initializing order control variables
    cam_names_read_ = 0;
    images_read_ = 0;
    corners_found_ = 0; 
    cams_initialized_ = 0;

}

void multiCamCalibration::run() {

    if (run_calib_flag) {

        if (mp4_) {
            read_cam_names_mp4();
            read_calib_imgs_mp4();
        } else if (mtiff_) {
            read_cam_names_mtiff();
            read_calib_imgs_mtiff();
        } else {
            read_cam_names();
            read_calib_imgs();
        }

        find_corners();
        
        if (refractive_) {
            initialize_cams();
            //initialize_cams_ref();
            write_BA_data_ref();
            run_BA_ref();
        } else {
            initialize_cams();
            write_BA_data();
            run_BA();
        }
        
        //calc_space_warp_factor();
        
        if (refractive_) {
            write_calib_results_ref();
        } else {
            write_calib_results();
        }
        
        // if (refractive_) {
        //     write_calib_results_matlab_ref();
        // } else {
        //     write_calib_results_matlab();
        // }
        
    }
    
    // if (load_results_flag) {
    //     load_calib_results();
    // }
    
    LOG(INFO)<<"\nCALIBRATION COMPLETE!\n";

}

void multiCamCalibration::read_cam_names() {

    DIR *dir;
    struct dirent *ent;
    dir = opendir(path_.c_str());

    // Reading camera names
    string temp_name;
    string dir1(".");
    string dir2("..");
    int i=1;

    LOG(INFO)<<"\nREADING CAMERA NAMES...\n\n";
    while (ent = readdir(dir)) {
        temp_name = ent->d_name;
        if (temp_name.compare(dir1)) {
            if (temp_name.compare(dir2)) {
                if (temp_name.compare(result_dir_)) {
                    //cout<<"Camera "<<i<<": "<<ent->d_name<<"\n";
                    cam_names_.push_back(ent->d_name);
                    i++;
                }
            }
        }
    }
    sort(cam_names_.begin(), cam_names_.end());
    
    for (int i=0; i<cam_names_.size(); i++) VLOG(1)<<"Camera "<<i+1<<": "<<cam_names_[i]<<endl;

    if (dummy_mode_) {
        int id;
        LOG(INFO)<<"Enter the center camera number: ";
        cin>>id;
        center_cam_id_ = id-1;
    } else {
        center_cam_id_ = 4;
    }

    num_cams_ = cam_names_.size();
    // refocusing_params_.num_cams = num_cams_;

    cam_names_read_ = 1;

}

void multiCamCalibration::read_cam_names_mtiff() {

    DIR *dir;
    struct dirent *ent;

    string dir1(".");
    string dir2("..");
    string temp_name;

    dir = opendir(path_.c_str());
    while(ent = readdir(dir)) {
        temp_name = ent->d_name;
        if (temp_name.compare(dir1)) {
            if (temp_name.compare(dir2)) {
                if (temp_name.compare(temp_name.size()-3,3,"tif") == 0) {
                    cam_names_.push_back(temp_name);
                }
            }
        }
    }
    
    sort(cam_names_.begin(), cam_names_.end());

    for (int i=0; i<cam_names_.size(); i++) VLOG(1)<<"Camera "<<i+1<<": "<<cam_names_[i]<<endl;

    if (dummy_mode_) {
        int id;
        LOG(INFO)<<"Enter the center camera number: ";
        cin>>id;
        center_cam_id_ = id-1;
    } else {
        center_cam_id_ = 4;
    }

    num_cams_ = cam_names_.size();
    // refocusing_params_.num_cams = num_cams_;

    cam_names_read_ = 1;

}

void multiCamCalibration::read_cam_names_mp4() {

    DIR *dir;
    struct dirent *ent;

    string dir1(".");
    string dir2("..");
    string temp_name;

    dir = opendir(path_.c_str());
    while(ent = readdir(dir)) {
        temp_name = ent->d_name;
        if (temp_name.compare(dir1)) {
            if (temp_name.compare(dir2)) {
                if (temp_name.compare(temp_name.size()-3,3,"MP4") == 0) {
                    cam_names_.push_back(temp_name);
                }
            }
        }
    }
    
    sort(cam_names_.begin(), cam_names_.end());

    for (int i=0; i<cam_names_.size(); i++) VLOG(1)<<"Camera "<<i+1<<": "<<cam_names_[i]<<endl;

    if (dummy_mode_) {
        int id;
        LOG(INFO)<<"Enter the center camera number: ";
        cin>>id;
        center_cam_id_ = id-1;
    } else {
        center_cam_id_ = 3;
    }

    num_cams_ = cam_names_.size();

    cam_names_read_ = 1;

}

void multiCamCalibration::read_calib_imgs() {

    if (!cam_names_read_) {
        LOG(WARNING)<<"Camera names have not been read yet! Call read_cam_names() first.";
    } else {

        DIR *dir;
        struct dirent *ent;
 
        string dir1(".");
        string dir2("..");
        string temp_name;
        string img_prefix = "";

        vector<string> img_names;
        Mat image;

        LOG(INFO)<<"\nREADING IMAGES...\n\n";

        for (int i=0; i<num_cams_; i++) {

            VLOG(1)<<"Camera "<<i+1<<" of "<<num_cams_<<"...";

            string path_tmp;
            vector<Mat> calib_imgs_sub;

            path_tmp = path_+cam_names_[i]+"/"+img_prefix;

            dir = opendir(path_tmp.c_str());
            while(ent = readdir(dir)) {
                temp_name = ent->d_name;
                if (temp_name.compare(dir1)) {
                    if (temp_name.compare(dir2)) {
                        string path_img = path_tmp+temp_name;
                        img_names.push_back(path_img);
                    }
                }
            }
        
            sort(img_names.begin(), img_names.end());
            for (int j=0; j<img_names.size(); j++) {
                if (i==0) {
                    VLOG(1)<<endl<<j<<": "<<img_names[j];
                }
                image = imread(img_names[j], 0);
                calib_imgs_sub.push_back(image);
            }
            img_names.clear();

            if (dummy_mode_) {
                if (i==0) {
                    int id;
                    LOG(INFO)<<endl<<"Enter the image with grid to be used to define origin: ";
                    cin>>id;
                    origin_image_id_ = id;
                }
            } else {
                if (i==0) {
                    origin_image_id_ = 4;
                }
            }

            calib_imgs_.push_back(calib_imgs_sub);
            path_tmp = "";

            LOG(INFO)<<"done!\n";
   
        }
 
        num_imgs_ = calib_imgs_[0].size();
        img_size_ = Size(calib_imgs_[0][0].cols, calib_imgs_[0][0].rows);
        // refocusing_params_.img_size = img_size_;

        LOG(INFO)<<"\nDONE READING IMAGES!\n\n";
        images_read_ = 1;

    }

}

void multiCamCalibration::read_calib_imgs_mtiff() {

    if (!cam_names_read_) {
        LOG(WARNING)<<"Camera names have not been read yet! Call read_cam_names_mtiff() first.";
    } else {

        string img_path;

        vector<TIFF*> tiffs;
        for (int i=0; i<cam_names_.size(); i++) {
            img_path = path_+cam_names_[i];
            TIFF* tiff = TIFFOpen(img_path.c_str(), "r");
            tiffs.push_back(tiff);
        }

        LOG(INFO)<<"Counting number of frames...";
        int dircount = 0;
        if (tiffs[0]) {
            do {
                dircount++;
            } while (TIFFReadDirectory(tiffs[0]));
        }
        LOG(INFO)<<"done! "<<dircount<<" frames found."<<endl<<endl;

        LOG(INFO)<<"Reading images..."<<endl;
        for (int i=0; i<cam_names_.size(); i++) {
        
            LOG(INFO)<<"Camera "<<i+1<<"...";

            vector<Mat> calib_imgs_sub;

            int frame=0;
            int count=0;
            int skip=skip_frames_;
            while (frame<dircount) {

                Mat img;
                uint32 c, r;
                size_t npixels;
                uint32* raster;
            
                TIFFSetDirectory(tiffs[i], frame);

                TIFFGetField(tiffs[i], TIFFTAG_IMAGEWIDTH, &c);
                TIFFGetField(tiffs[i], TIFFTAG_IMAGELENGTH, &r);
                npixels = r * c;
                raster = (uint32*) _TIFFmalloc(npixels * sizeof (uint32));
                if (raster != NULL) {
                    if (TIFFReadRGBAImageOriented(tiffs[i], c, r, raster, ORIENTATION_TOPLEFT, 0)) {
                        img.create(r, c, CV_32F);
                        for (int i=0; i<r; i++) {
                            for (int j=0; j<c; j++) {
                                img.at<float>(i,j) = TIFFGetR(raster[i*c+j])/255.0;
                            }
                        }
                    }
                    _TIFFfree(raster);
                }

                img *= 255;
                Mat img1; img.convertTo(img1, CV_8U);
                calib_imgs_sub.push_back(img1);
                count++;
            
                frame += skip;

            }

            calib_imgs_.push_back(calib_imgs_sub);
            LOG(INFO)<<"done! "<<count<<" frames read."<<endl;

        }

        num_imgs_ = calib_imgs_[0].size();
        img_size_ = Size(calib_imgs_[0][0].cols, calib_imgs_[0][0].rows);

        LOG(INFO)<<"\nDONE READING IMAGES!\n\n";
        images_read_ = 1;

    }

}

void multiCamCalibration::read_calib_imgs_mp4() {

    for (int i=0; i<cam_names_.size(); i++) {

        string file_path = path_+cam_names_[i];
        LOG(INFO)<<"Opening "<<file_path;
        VideoCapture cap(file_path); // open the default camera
        if(!cap.isOpened())  // check if we succeeded
            LOG(FATAL)<<"Could not open " + file_path;

        int total_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
        VLOG(1)<<"Total frames: "<<total_frames;

        int count = 0;
        int counter = 0;
        Mat frame, frame2;

        // skip initial frames
        while(counter<start_frame_-1) {
            cap >> frame; counter++;
        }
        VLOG(3)<<counter<<" frames skipped";

        vector<Mat> calib_imgs_sub;
        while(counter<=end_frame_) {
            // Mat frame;
            cap >> frame; counter++;
            VLOG(3)<<counter<<"\'th frame read";
            cvtColor(frame, frame2, CV_BGR2GRAY);
            frame2.convertTo(frame2, CV_8U);
            calib_imgs_sub.push_back(frame2.clone()); // store frame
            count++;

            VLOG(3)<<"Skipping...";
            int skipped = 0;
            while(skipped<skip_frames_) {
                cap >> frame; counter++;
                skipped++;
            }
            VLOG(3)<<"Counter at: "<<counter;
        }

        calib_imgs_.push_back(calib_imgs_sub);
        LOG(INFO)<<"done! "<<count<<" frames read."<<endl;

    }

    num_imgs_ = calib_imgs_[0].size();
    img_size_ = Size(calib_imgs_[0][0].cols, calib_imgs_[0][0].rows);

    LOG(INFO)<<"\nDONE READING IMAGES!\n\n";
    images_read_ = 1;

}

// TODO: add square grid correction capability again
void multiCamCalibration::find_corners() {

    if (!images_read_) {
        LOG(WARNING)<<"Images have have not been read yet! Call read_calib_imgs() or read_calib_imgs_mtiff first.";
    } else {

        vector< vector<Point2f> > corner_points;
        vector<Point2f> points;
        Mat scene, scene_gray, scene_drawn;

        Mat_<double> found_mat = Mat_<double>::zeros(num_cams_, calib_imgs_[0].size());
    
        LOG(INFO)<<"\nFINDING CORNERS...\n\n";
        for (int i=0; i<num_cams_; i++) {
        
            LOG(INFO)<<"Camera "<<i+1<<" of "<<num_cams_<<"...";

            int not_found=0;
            vector<Mat> imgs_temp = calib_imgs_[i];
            for (int j=0; j<imgs_temp.size(); j++) {

                //equalizeHist(imgs_temp[j], scene);
                scene = imgs_temp[j]; //*255.0;
                //scene.convertTo(scene, CV_8U);
                bool found = findChessboardCorners(scene, grid_size_, points, CV_CALIB_CB_ADAPTIVE_THRESH);
            
                if (found) {
                    //cvtColor(scene, scene_gray, CV_RGB2GRAY);
                    cornerSubPix(scene, points, Size(10, 10), Size(-1, -1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
                    corner_points.push_back(points);
                    found_mat(i,j) = 1;

                    if (show_corners_flag) {
                        scene_drawn = scene;
                        cvtColor(scene_drawn, scene_drawn, CV_GRAY2RGB);
                        drawChessboardCorners(scene_drawn, grid_size_, points, found);
                        namedWindow("Pattern", CV_WINDOW_NORMAL);
                        imshow("Pattern", scene_drawn);
                        waitKey(0);
                        cvDestroyWindow("Pattern");
                    }
                
                } else {
                    not_found++;
                
                    points.clear(); points.push_back(Point2f(0,0));
                    corner_points.push_back(points);

                    if (show_corners_flag) {
                        namedWindow("Pattern not found!", CV_WINDOW_AUTOSIZE);
                        imshow("Pattern not found!", scene);
                    }
                    LOG(WARNING)<<"pattern not found!\n";
                }

            }

            all_corner_points_raw_.push_back(corner_points);
            corner_points.clear();
            LOG(INFO)<<"done! Corners not found in "<<not_found<<" image(s)";
            if (not_found==imgs_temp.size())
                LOG(WARNING)<<"Need more images in which corners are found!";

        }

        // Getting rid of unfound corners
        Mat_<double> a = Mat_<double>::ones(1,num_cams_);
        Mat b = (a*found_mat)/num_cams_;
        int sum=0;
        for (int i=0; i<b.cols; i++)
            sum += floor(b.at<double>(0,i));

        if (sum<3)
            LOG(WARNING)<<"Not enough pictures from each camera with found corners!";

        for (int i=0; i<num_cams_; i++) {
            for (int j=0; j<b.cols; j++) {
                if (b.at<double>(0,j)==1.0) {
                    corner_points.push_back(all_corner_points_raw_[i][j]);
                }
            }
            all_corner_points_.push_back(corner_points);
            corner_points.clear();
        }

        get_grid_size_pix();
        pix_per_phys_ = grid_size_pix_/grid_size_phys_;
        // refocusing_params_.scale = pix_per_phys_;

        LOG(INFO)<<"\nCORNER FINDING COMPLETE!\n\n";
        corners_found_ = 1;

    }

}

void multiCamCalibration::initialize_cams_ref() {

    int cam=0;
    double z = 27.5;

    int points_per_img = grid_size_.width*grid_size_.height;
    int num_imgs = calib_imgs_[cam].size();

    double* points = new double[3*num_imgs*points_per_img];

    double* camera = new double[9];
    for (int j=0; j<9; j++) {
        camera[j] = 0;
    }
    camera[6] = 9000;
    camera[5] = -1000+z;

    int origin=0;
    for (int i=0; i<num_imgs; i++) {
        for (int j=0; j<grid_size_.height; j++) {
            for (int k=0; k<grid_size_.width; k++) {
                if (i==origin) {
                    points[3*(i*points_per_img+j*grid_size_.width+k)] = grid_size_phys_*(k - (grid_size_.width-1)*0.5);
                    points[3*(i*points_per_img+j*grid_size_.width+k)+1] = grid_size_phys_*(j - (grid_size_.height-1)*0.5);
                    points[3*(i*points_per_img+j*grid_size_.width+k)+2] = z;
                } else {
                    points[3*(i*points_per_img+j*grid_size_.width+k)] = (double(rand()%1)-0.5)*grid_size_phys_*grid_size_.width;
                    points[3*(i*points_per_img+j*grid_size_.width+k)+1] = (double(rand()%1)-0.5)*grid_size_phys_*grid_size_.height;
                    points[3*(i*points_per_img+j*grid_size_.width+k)+2] = z; // TODO: CHANGE?
                }
            }
        }
    }

    double* planes = new double[num_imgs*4];
    for (int i=0; i<num_imgs; i++) {
        if (i==origin) {
            planes[i*4] = 0;
            planes[i*4+1] = 0;
            planes[i*4+2] = 1;
            planes[i*4+3] = 0;
        } else {
            planes[i*4] = 1;
            planes[i*4+1] = 1;
            planes[i*4+2] = 1;
            planes[i*4+3] = 1;
        }
    }

    ceres::Problem problem;
    for (int i=0; i<num_imgs; i++) {
        for (int j=0; j<points_per_img; j++) {

            ceres::CostFunction* cost_function1 =
                new ceres::NumericDiffCostFunction<refractiveReprojectionError, ceres::CENTRAL, 2, 9, 3>
                (new refractiveReprojectionError(all_corner_points_[cam][i][j].x,
                                                 all_corner_points_[cam][i][j].y,
                                                 img_size_.width*0.5, img_size_.height*0.5, 
                                                 1, 5.0, 1.0, 1.0, 1.3, 0 ));
            
            problem.AddResidualBlock(cost_function1,
                                     NULL,
                                     camera, points + 3*(i*points_per_img + j));

            if (i==origin)
                problem.SetParameterBlockConstant(points + 3*(i*points_per_img + j));
       
            ceres::CostFunction* cost_function2 =
                new ceres::AutoDiffCostFunction<planeError, 1, 3, 4>
                (new planeError(1));

            problem.AddResidualBlock(cost_function2,
                                     NULL,
                                     points + 3*(i*points_per_img + j), planes + i*4);

        /*

          problem.SetParameterBlockConstant(ba_problem.mutable_point_for_observation(i));

        */

        }
    }
    
    // Adding constraint for grid physical size
    for (int i=0; i<num_imgs; i++) {
        
        ceres::CostFunction* cost_function3 = 
            new ceres::NumericDiffCostFunction<gridPhysSizeError, ceres::CENTRAL, 1, 3, 3, 3>
            (new gridPhysSizeError(grid_size_phys_, grid_size_.width, grid_size_.height));
        
        problem.AddResidualBlock(cost_function3,
                                 NULL,
                                 points + 3*(i*points_per_img) + 0,
                                 points + 3*(i*points_per_img) + 3*(grid_size_.width-1),
                                 points + 3*(i*points_per_img) + 3*(grid_size_.width*(grid_size_.height-1)));

    }
    

    LOG(INFO)<<"Initializing Cam"<<endl;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;//DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = refractive_max_iterations;
    
    // int threads = omp_get_num_procs();
    // options.num_threads = threads;
    // LOG(INFO)<<"\nSolver using "<<threads<<" threads.\n\n";

    options.gradient_tolerance = 1E-12;
    options.function_tolerance = 1E-8;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    LOG(INFO)<<summary.FullReport()<<"\n";
    
    for (int i=0; i<9; i++) {
        LOG(INFO)<<camera[i]<<endl;
    }

}

void multiCamCalibration::initialize_cams() {

    if (!corners_found_) {

    } else {

        vector<Point3f> pattern_points_single;
        vector< vector<Point3f> > pattern_points;
        float xshift = -(grid_size_.width-1)/2.0;
        float yshift = -(grid_size_.height-1)/2.0;
        for (int i=0; i<grid_size_.height; i++) {
            for (int j=0; j<grid_size_.width; j++) {
                pattern_points_single.push_back(Point3f( 5*(float(j)+xshift), 5*(float(i)+yshift), 0.0f));
            }
        }
        for (int i=0; i<num_cams_; i++) {
            for (int j=0; j<all_corner_points_[i].size(); j++) {
                pattern_points.push_back(pattern_points_single);
            }
            all_pattern_points_.push_back(pattern_points);
            pattern_points.clear();
        }

        LOG(INFO)<<"\nINITIALIZING CAMERAS...\n\n";

        for (int i=0; i<num_cams_; i++) {

            LOG(INFO)<<"Calibrating camera "<<i+1<<"...";
            Mat_<double> A = Mat_<double>::zeros(3,3); 
            Mat_<double> dist_coeff;

            vector<Mat> rvec, tvec;

            A(0,0) = init_f_value_;
            A(1,1) = init_f_value_;
            A(0,2) = img_size_.width*0.5;
            A(1,2) = img_size_.height*0.5;
            A(2,2) = 1;
        
            //vector<Mat> rvec, tvec;
            calibrateCamera(all_pattern_points_[i], all_corner_points_[i], img_size_, A, dist_coeff, rvec, tvec, CV_CALIB_USE_INTRINSIC_GUESS|CV_CALIB_FIX_PRINCIPAL_POINT|CV_CALIB_FIX_ASPECT_RATIO);
            //calibrateCamera(all_pattern_points_, all_corner_points_[i], img_size_, A, dist_coeff, rvec, tvec, CV_CALIB_FIX_ASPECT_RATIO);
            LOG(INFO)<<"done!\n";

            LOG(INFO)<<A<<endl;

            cameraMats_.push_back(A);
            dist_coeffs_.push_back(dist_coeff);
            rvecs_.push_back(rvec);
            tvecs_.push_back(tvec);

        }

        LOG(INFO)<<"\nCAMERA INITIALIZATION COMPLETE!\n\n";
        cams_initialized_ = 1;

    }

}

// TODO: Not general and just not required here. Needs to go.
void multiCamCalibration::write_BA_data() {

    LOG(INFO)<<"\nWRITING POINT DATA TO FILE...";
    
    ofstream file;
    file.open(ba_file_.c_str());
    if (!file)
        LOG(FATAL)<<"Unable to open file "<<ba_file_<<" to write data!";

    int imgs_per_cam = all_corner_points_[0].size();
    int points_per_img = all_corner_points_[0][0].size();
    int num_points = imgs_per_cam*points_per_img;
    int observations = num_points*num_cams_;

    // Calibration set configuration parameters
    file<<num_cams_<<"\t"<<imgs_per_cam<<"\t"<<num_points<<"\t"<<observations<<"\n";

    // Observation points i.e. detected corners
    for (int j=0; j<imgs_per_cam; j++) {
        for (int k=0; k<points_per_img; k++) {
            for (int i=0; i<num_cams_; i++) {
                file<<i<<"\t"<<j<<"\t"<<(j*points_per_img)+k<<"\t"<<all_corner_points_[i][j][k].x<<"\t"<<all_corner_points_[i][j][k].y<<endl;
            }
        }
    }

    double param = 0;
    
    // R, T, K and distortion coeffs for each camera
    for (int i=0; i<num_cams_; i++) {
        for (int j=0; j<3; j++) {
            file<<rvecs_[i][0].at<double>(0,j)<<"\t";
        }   
        for (int j=0; j<3; j++) {
            file<<tvecs_[i][0].at<double>(0,j)<<"\t";
        }
        file<<cameraMats_[i].at<double>(0,0)<<"\t";
        //for (int j=0; j<2; j++) file<<dist_coeffs[i].at<double>(0,j)<<"\n";
        file<<0<<"\t"<<0<<endl;
    }
    
    double width = 6*5;
    double height = 5*5;
    // add back projected point guesses here
    double z = 0;
    int op1 = (origin_image_id_)*grid_size_.width*grid_size_.height;
    int op2 = op1+grid_size_.width-1;
    int op3 = op1+(grid_size_.width*(grid_size_.height-1));
    const_points_.push_back(op1);
    const_points_.push_back(op2);
    const_points_.push_back(op3);
    
    // Initial values for grid points in 3D
    for (int i=0; i<num_points; i++) {
        if (i==op1) {
            file<<double(-grid_size_.width*grid_size_phys_*0.5)<<"\t";
            file<<double(-grid_size_.height*grid_size_phys_*0.5)<<"\t";
            file<<z<<"\t"<<endl;
        } else if (i==op2) {
            file<<double(grid_size_.width*grid_size_phys_*0.5)<<"\t";
            file<<double(-grid_size_.height*grid_size_phys_*0.5)<<"\t";
            file<<z<<"\t"<<endl;
        } else if (i==op3) {
            file<<double(-grid_size_.width*grid_size_phys_*0.5)<<"\t";
            file<<double(grid_size_.height*grid_size_phys_*0.5)<<"\t";
            file<<z<<"\t"<<endl;
        } else {
            file<<(double(rand()%50)-(width*0.5))<<"\t";
            file<<(double(rand()%50)-(height*0.5))<<"\t";
            file<<(rand()%50)+z<<"\t"<<endl;
            // file<<0<<"\t"<<endl;
        }
    }
    

    /*/ Initializing values for grid points in 3D as all zero
    for (int i=0; i<num_points; i++) {
        file<<0<<"\t"<<0<<"\t"<<0<<endl;
    }
    */

    /*
    for (int k=0; k<6; k++) {
    for (int i=0; i<5; i++) {
        for (int j=0; j<5; j++) {
            file<<(5*i)-25<<endl<<(5*j)-20<<endl<<(5*k)-20<<endl;
            //file2<<(5*i)-25<<endl<<(5*j)-20<<endl<<(5*k)-20<<endl;
        }
    }
    }
    */

    param = 1;
    for (int i=0; i<imgs_per_cam; i++) {
        if (i==origin_image_id_) {
            file<<0<<"\t"<<0<<"\t"<<1<<"\t"<<0<<endl;
        } else {
            for (int j=0; j<4; j++) {
                file<<param<<"\t";
            }
            file<<endl;
        }
    }

    file.close();
    LOG(INFO)<<"DONE!\n";

}

// Write BA data for refractive calibration
void multiCamCalibration::write_BA_data_ref() {

    LOG(INFO)<<"\nWRITING POINT DATA TO FILE...";
    ofstream file;
    file.open(ba_file_.c_str());

    int imgs_per_cam = all_corner_points_[0].size();
    int points_per_img = all_corner_points_[0][0].size();
    int num_points = imgs_per_cam*points_per_img;
    int observations = num_points*num_cams_;

    file<<num_cams_<<"\t"<<imgs_per_cam<<"\t"<<num_points<<"\t"<<observations<<"\n";
    for (int j=0; j<imgs_per_cam; j++) {
        for (int k=0; k<points_per_img; k++) {
            for (int i=0; i<num_cams_; i++) {
                file<<i<<"\t"<<j<<"\t"<<(j*points_per_img)+k<<"\t"<<all_corner_points_[i][j][k].x<<"\t"<<all_corner_points_[i][j][k].y<<endl;
            }
        }
    }
    double param = 0;
    for (int i=0; i<num_cams_; i++) {
        for (int j=0; j<3; j++) {
            file<<rvecs_[i][0].at<double>(0,j)<<"\t";
        }
        
        for (int j=0; j<3; j++) {
            file<<tvecs_[i][0].at<double>(0,j)<<"\t";
        }
        
        //file<<cameraMats_[i].at<double>(0,0)<<"\t";
        file<<9000.0<<"\t";
        //for (int j=0; j<2; j++) file<<dist_coeffs[i].at<double>(0,j)<<"\n";
        file<<0<<"\t"<<0<<endl;
    }
    
    param = 1;
    for (int i=0; i<imgs_per_cam; i++) {
            
        for (int j=0; j<3; j++) {
            file<<0<<"\t";
        }

        file<<0<<"\t";
        file<<0<<"\t";
        file<<0<<"\t";

        file<<endl;

    }

    // thickness and refractive indices
    file<<5.0<<endl;
    file<<1.0<<endl;
    file<<1.0<<endl;
    file<<1.3<<endl;
    //file<<100-72.5<<endl;
    file<<-100.0<<endl;

    file.close();
    LOG(INFO)<<"DONE!\n";

}

void multiCamCalibration::run_BA() {

    run_BA_pinhole(ba_problem_, ba_file_, img_size_, const_points_);

}

void multiCamCalibration::run_BA_ref() {

    run_BA_refractive(ba_problem_ref_, ba_file_, img_size_, const_points_);

}

// Pinhole bundle adjustment function
double multiCamCalibration::run_BA_pinhole(baProblem &ba_problem, string ba_file, Size img_size, vector<int> const_points) {

    LOG(INFO)<<"\nRUNNING PINHOLE BUNDLE ADJUSTMENT TO CALIBRATE CAMERAS...\n";
    //google::InitGoogleLogging(argv);
    
    if (!ba_problem.LoadFile(ba_file.c_str()))
        LOG(FATAL)<<"Unable to open file "<<ba_file<<" to read data!";

    ba_problem.cx = img_size.width*0.5;
    ba_problem.cy = img_size.height*0.5;
    
    // Create residuals for each observation in the bundle adjustment problem. The
    // parameters for cameras and points are added automatically.
    ceres::Problem problem;

    for (int i=0; i<ba_problem.num_observations(); i++) {

        // Each Residual block takes a point and a camera as input and outputs a 2
        // dimensional residual. Internally, the cost function stores the observed
        // image location and compares the reprojection against the observation.
        ceres::CostFunction* cost_function1 =
            new ceres::AutoDiffCostFunction<pinholeReprojectionError, 2, 9, 3>
            (new pinholeReprojectionError(ba_problem.observations()[2 * i + 0],
                                          ba_problem.observations()[2 * i + 1],
                                          ba_problem.cx, ba_problem.cy, ba_problem.num_cameras()));
        
        problem.AddResidualBlock(cost_function1,
                                 NULL,
                                 ba_problem.mutable_camera_for_observation(i),
                                 ba_problem.mutable_point_for_observation(i));

        ceres::CostFunction* cost_function2 =
            new ceres::AutoDiffCostFunction<planeError, 1, 3, 4>
            (new planeError(ba_problem.num_cameras()));

        problem.AddResidualBlock(cost_function2,
                                 NULL,
                                 ba_problem.mutable_point_for_observation(i),
                                 ba_problem.mutable_plane_for_observation(i));

    }
    
    int gridx = grid_size_.width;
    int gridy = grid_size_.height;

    // Adding constraint for grid physical size
    for (int i=0; i<ba_problem.num_planes(); i++) {

        ceres::CostFunction* cost_function3 = 
            new ceres::NumericDiffCostFunction<gridPhysSizeError, ceres::CENTRAL, 1, 3, 3, 3>
            (new gridPhysSizeError(grid_size_phys_, gridx, gridy));

        problem.AddResidualBlock(cost_function3,
                                 NULL,
                                 ba_problem.mutable_points() + 3*gridx*gridy*i + 0,
                                 ba_problem.mutable_points() + 3*gridx*gridy*i + 3*(gridx-1),
                                 ba_problem.mutable_points() + 3*gridx*gridy*i + 3*(gridx*(gridy-1)));

    }
    
    // Fixing a plane to xy plane
    
    int i=origin_image_id_;

    // ceres::CostFunction* cost_function4 = 
    //     new ceres::NumericDiffCostFunction<zError, ceres::CENTRAL, 1, 3, 3, 3, 3>
    //     (new zError());
    // problem.AddResidualBlock(cost_function4,
    //                          NULL,
    //                          ba_problem.mutable_points() + 3*gridx*gridy*i + 0,
    //                          ba_problem.mutable_points() + 3*gridx*gridy*i + 3*(gridx-1),
    //                          ba_problem.mutable_points() + 3*gridx*gridy*i + 3*(gridx*(gridy-1)),
    //                          ba_problem.mutable_points() + 3*gridx*gridy*i + 3*(gridx*gridy-1)); 

    ceres::CostFunction* cost_function4 = 
        new ceres::NumericDiffCostFunction<zError2, ceres::CENTRAL, 1, 4>
        (new zError2());
    problem.AddResidualBlock(cost_function4,
                             NULL,
                             ba_problem.mutable_planes() + 4*i); 

    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = pinhole_max_iterations;
    
    // int threads = omp_get_num_procs();
    // options.num_threads = threads;
    // LOG(INFO)<<"\nSolver using "<<threads<<" threads.\n\n";

    options.gradient_tolerance = 1E-12;
    options.function_tolerance = 1E-8;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    LOG(INFO)<<summary.FullReport()<<"\n";
    LOG(INFO)<<"BUNDLE ADJUSTMENT COMPLETE!\n\n";

    total_error_ = double(summary.final_cost);
    avg_error_ = total_error_/ba_problem.num_observations(); // TODO: probably wrong but whatever
    LOG(INFO)<<"Total Final Error:\t"<<total_error_<<endl;
    LOG(INFO)<<"Average Final Error:\t"<<avg_error_<<endl;

}

// Refractivee bundle adjustment function
double multiCamCalibration::run_BA_refractive(baProblem_plane &ba_problem, string ba_file, Size img_size, vector<int> const_points) {

    LOG(INFO)<<"\nRUNNING REFRACTIVE BUNDLE ADJUSTMENT TO CALIBRATE CAMERAS...\n";
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
    for (int i=0; i<ba_problem.num_observations(); i++) {

        // Each Residual block takes a point and a camera as input and outputs a 2
        // dimensional residual. Internally, the cost function stores the observed
        // image location and compares the reprojection against the observation.
        ceres::CostFunction* cost_function1 =
            new ceres::NumericDiffCostFunction<refractiveReprojError, ceres::CENTRAL, 2, 9, 6>
            (new refractiveReprojError(ba_problem.observations()[2 * i + 0],
                                       ba_problem.observations()[2 * i + 1],
                                       ba_problem.cx, ba_problem.cy, 
                                       ba_problem.num_cameras(),
                                       ba_problem.t(), ba_problem.n1(), ba_problem.n2(), ba_problem.n3(), ba_problem.z0(),
                                       grid_size_.width, grid_size_.height, grid_size_phys_, 
                                       ba_problem.point_index()[i], ba_problem.plane_index()[i] ));
        
        problem.AddResidualBlock(cost_function1,
                                 NULL,
                                 ba_problem.mutable_camera_for_observation(i),
                                 ba_problem.mutable_plane_for_observation(i));
        
        /*

          problem.SetParameterBlockConstant(ba_problem.mutable_point_for_observation(i));

        */

    }

    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100; //refractive_max_iterations;
    
    // int threads = omp_get_num_procs();
    // options.num_threads = threads-2;
    // LOG(INFO)<<"\nSolver using "<<threads<<" threads.\n\n";

    options.gradient_tolerance = 1E-12;
    options.function_tolerance = 1E-8;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    LOG(INFO)<<summary.FullReport()<<"\n";
    LOG(INFO)<<"BUNDLE ADJUSTMENT COMPLETE!\n\n";

    total_error_ = double(summary.final_cost);
    avg_error_ = total_error_/ba_problem.num_observations();
    LOG(INFO)<<"Total Final Error:\t"<<total_error_<<endl;
    LOG(INFO)<<"Average Final Error:\t"<<avg_error_<<endl;

}

/*
// TODO: CHANGE THE CODE SO THAT THIS FUNCTION IS NOT NEEDED AND THE PHYSICAL
//       GRID SIZE IS TAKEN INTO ACCOUNT DURING BUNDLE ADJUSTMENT
void multiCamCalibration::calc_space_warp_factor() {

    double* world_points;
    if (refractive_) {
        world_points = ba_problem_ref_.mutable_points();
    } else {
        world_points = ba_problem_.mutable_points();
    }

    double total_dist = 0;
    double dist_prev;
    int ind1, ind2;

    int count=0;
    for (int i=0; i<num_imgs_; i++) {
        for (int j=0; j<grid_size_.height-1; j++) {
            for (int k=0; k<grid_size_.width-1; k++) {

                ind1 = (i*grid_size_.width*grid_size_.height)+(j*grid_size_.width)+k;
                
                ind2 = ind1+1;
                dist_prev = total_dist;
                total_dist += sqrt( pow(world_points[3*ind2]-world_points[3*ind1],2)+pow(world_points[3*ind2+1]-world_points[3*ind1+1],2)+pow(world_points[3*ind2+2]-world_points[3*ind1+2],2) );
                //LOG(INFO)<<total_dist-dist_prev<<endl;
                
                ind2 = ind1+grid_size_.width;
                dist_prev = total_dist;
                total_dist += sqrt( pow(world_points[3*ind2]-world_points[3*ind1],2)+pow(world_points[3*ind2+1]-world_points[3*ind1+1],2)+pow(world_points[3*ind2+2]-world_points[3*ind1+2],2) );
                //LOG(INFO)<<total_dist-dist_prev<<endl;
                
                count += 2;

            }
        }
    }

    warp_factor_ = (total_dist/count)/grid_size_phys_;
    refocusing_params_.warp_factor = warp_factor_;

}
*/

void multiCamCalibration::write_calib_results() {

    char choice;
    LOG(INFO)<<"\nDo you wish to save the calibration results (y/n)?: ";
    cin>>choice;

    if (choice==89 || choice==121) {
        
        string newPath = path_ + result_dir_ + "/";
        if (!dirExists(newPath)) {
            LOG(INFO)<<"\'calibration_results\' directory does not exist..."<<endl;
            mkdir(newPath.c_str(), S_IRWXU);
            LOG(INFO)<<"directory created!"<<endl;
        }

        time_t timer;
        time(&timer);
        char time_stamp[15];
        sprintf(time_stamp, "%.15ld", timer);
        string time_stamp_str(time_stamp);
        result_file_ = newPath + "results_"+time_stamp_str+".txt";
        results_just_saved_flag = 1;
        
        struct tm * timeinfo;
        time(&timer);
        timeinfo = localtime(&timer);
        char time_stamp_hr[50];
        sprintf(time_stamp_hr, "Pinhole Calibration performed: %s", asctime(timeinfo));
        string time_stamp_hr_str(time_stamp_hr);

        double* camera_params = ba_problem_.mutable_cameras();

        ofstream file;
        file.open(result_file_.c_str());

        // WRITING DATA TO RESULTS FILE
        
        file<<time_stamp_hr_str;
        file<<avg_error_<<endl;
        file<<img_size_.width<<"\t"<<img_size_.height<<"\t"<<pix_per_phys_<<endl;
        file<<num_cams_<<endl;

        Mat_<double> rvec = Mat_<double>::zeros(1,3);
        Mat_<double> tvec = Mat_<double>::zeros(3,1);
        Mat_<double> K = Mat_<double>::zeros(3,3);
        Mat_<double> dist = Mat_<double>::zeros(1,2);
        Mat R, Rt, C;

        for (int i=0; i<num_cams_; i++) {

            for (int j=0; j<3; j++) {
                rvec(0,j) = camera_params[(i*num_cams_)+j];
                tvec(j,0) = camera_params[(i*num_cams_)+j+3];
            }

            Rodrigues(rvec, R);
            
            // refocusing_params_.cam_names.push_back(cam_names_[i]);
            
            K(0,0) = camera_params[(i*num_cams_)+6];
            K(1,1) = camera_params[(i*num_cams_)+6];
            K(0,2) = img_size_.width*0.5;
            K(1,2) = img_size_.height*0.5;
            K(2,2) = 1;

            transpose(R, Rt);
            C = -Rt*tvec;

            rVecs_.push_back(R.clone());
            tVecs_.push_back(tvec.clone());
            K_mats_.push_back(K.clone());
            cVecs_.push_back(C.clone());

        }
        
        // Mat_<double> P_u = Mat_<double>::zeros(3,4);
        Mat_<double> P = Mat_<double>::zeros(3,4);
        Mat_<double> rmean = Mat_<double>::zeros(3,3);
        matrixMean(rVecs_, rmean);
        Mat rmean_t;
        transpose(rmean, rmean_t);

        for (int i=0; i<num_cams_; i++) {

            R = rVecs_[i]*rmean_t;
            // R = rVecs_[i];

            for (int j=0; j<3; j++) {
                for (int k=0; k<3; k++) {
                    // P_u.at<double>(j,k) = rVecs_[i].at<double>(j,k);
                    P.at<double>(j,k) = R.at<double>(j,k);
                }
                // P_u.at<double>(j,3) = tVecs_[i].at<double>(0,j);
                P.at<double>(j,3) = tVecs_[i].at<double>(0,j);
            }
            
            // P_u = K*P_u;
            P = K*P;
            // // refocusing_params_.P_mats_u.push_back(P_u.clone());
            // refocusing_params_.P_mats.push_back(P.clone());

            // Writing camera names and P matrices to file
            file<<cam_names_[i]<<endl;
            for (int j=0; j<3; j++) {
                for (int k=0; k<3; k++) {
                    // file<<P_u.at<double>(j,k)<<"\t";
                }
                // file<<P_u.at<double>(j,3)<<endl;
            }
            for (int j=0; j<3; j++) {
                for (int k=0; k<3; k++) {
                    file<<P.at<double>(j,k)<<"\t";
                }
                file<<P.at<double>(j,3)<<endl;
            }
            for (int j=0; j<3; j++) {
                file<<cVecs_[i].at<double>(j,1)<<"\t";
            }
            file<<endl;
        }
        
        file<<refractive_<<endl;
        if(refractive_) {
            //file<<geom[0]<<"\t"<<geom[1]<<"\t"<<geom[2]<<"\t"<<geom[3]<<"\t"<<geom[4]<<endl;
        }

        file.close();

        LOG(INFO)<<"\nCalibration results saved to file: "<<result_file_<<endl;

    }

}

// TODO: Convert P mats to ones that can be used for refractive reprojection
void multiCamCalibration::write_calib_results_ref() {

    char choice;
    LOG(INFO)<<"\nDo you wish to save the calibration results (y/n)?: ";
    cin>>choice;

    if (choice==89 || choice==121) {
        
        string newPath = path_ + result_dir_ + "/";
        if (!dirExists(newPath)) {
            LOG(INFO)<<"\'calibration_results\' directory does not exist..."<<endl;
            mkdir(newPath.c_str(), S_IRWXU);
            LOG(INFO)<<"directory created!"<<endl;
        }

        time_t timer;
        time(&timer);
        char time_stamp[15];
        sprintf(time_stamp, "%.15ld", timer);
        string time_stamp_str(time_stamp);
        result_file_ = newPath + "results_"+time_stamp_str+".txt";
        results_just_saved_flag = 1;
        
        struct tm * timeinfo;
        time(&timer);
        timeinfo = localtime(&timer);
        char time_stamp_hr[50];
        sprintf(time_stamp_hr, "Refractive Calibration performed: %s", asctime(timeinfo));
        string time_stamp_hr_str(time_stamp_hr);

        double* camera_params = ba_problem_ref_.mutable_cameras();

        ofstream file;
        file.open(result_file_.c_str());

        // WRITING DATA TO RESULTS FILE
        
        file<<time_stamp_hr_str<<endl;
        file<<avg_reproj_error_<<endl;
        file<<num_cams_<<endl;
        file<<img_size_.width<<"\t"<<img_size_.height<<"\t"<<pix_per_phys_;

        Mat_<double> rvec = Mat_<double>::zeros(1,3);
        Mat_<double> tvec = Mat_<double>::zeros(3,1);
        Mat_<double> K = Mat_<double>::zeros(3,3);
        Mat_<double> dist = Mat_<double>::zeros(1,2);
        Mat R;

        for (int i=0; i<num_cams_; i++) {

            for (int j=0; j<3; j++) {
                rvec(0,j) = camera_params[(i*num_cams_)+j];
                tvec(j,0) = camera_params[(i*num_cams_)+j+3];
            }

            Rodrigues(rvec, R);
            
            // refocusing_params_.cam_names.push_back(cam_names_[i]);
            
            K(0,0) = camera_params[(i*num_cams_)+6];
            K(1,1) = camera_params[(i*num_cams_)+6];
            K(0,2) = img_size_.width*0.5;
            K(1,2) = img_size_.height*0.5;
            K(2,2) = 1;

            rVecs_.push_back(R.clone());
            tVecs_.push_back(tvec.clone());
            K_mats_.push_back(K.clone());

        }
        
        //Mat_<double> P_u = Mat_<double>::zeros(3,4);
        Mat_<double> P = Mat_<double>::zeros(3,4);
        Mat_<double> rmean = Mat_<double>::zeros(3,3);
        matrixMean(rVecs_, rmean);
        Mat rmean_t;
        transpose(rmean, rmean_t);
        for (int i=0; i<num_cams_; i++) {

            R = rVecs_[i]*rmean_t;
    
            for (int j=0; j<3; j++) {
                for (int k=0; k<3; k++) {
                    //P_u.at<double>(j,k) = rVecs_[i].at<double>(j,k);
                    P.at<double>(j,k) = R.at<double>(j,k);
                }
                //P_u.at<double>(j,3) = tVecs_[i].at<double>(0,j);
                P.at<double>(j,3) = tVecs_[i].at<double>(0,j);
            }
            
            //P_u = K*P_u;
            P = K*P;
            // // refocusing_params_.P_mats_u.push_back(P_u.clone());
            // refocusing_params_.P_mats.push_back(P.clone());

            // Writing camera names and P matrices to file
            file<<cam_names_[i]<<endl;
            for (int j=0; j<3; j++) {
                for (int k=0; k<3; k++) {
                    //file<<P_u.at<double>(j,k)<<"\t";
                }
                //file<<P_u.at<double>(j,3)<<endl;
            }
            for (int j=0; j<3; j++) {
                for (int k=0; k<3; k++) {
                    file<<P.at<double>(j,k)<<"\t";
                }
                file<<P.at<double>(j,3)<<endl;
            }

        }
        
        file<<refractive_<<endl;
        if(refractive_) {
            //file<<geom[0]<<"\t"<<geom[1]<<"\t"<<geom[2]<<"\t"<<geom[3]<<"\t"<<geom[4]<<endl;
        }

        file.close();

        LOG(INFO)<<"\nCalibration results saved to file: "<<result_file_<<endl;

    }

}
/* Probably not needed anymore
void multiCamCalibration::load_calib_results() {

    ifstream file;
    
    // add stuff to read from folder for single and multiple result file cases

    DIR *dir;
    struct dirent *ent;
    dir = opendir((path_+result_dir_+"/").c_str());

    string temp_name;
    string dir1(".");
    string dir2("..");
    vector<string> result_files;
    int i=0;

    LOG(INFO)<<"\nLOOKING FOR CALIBRATION RESULT FILES...\n\n";
    while (ent = readdir(dir)) {
        temp_name = ent->d_name;
        if (temp_name.compare(dir1)) {
            if (temp_name.compare(dir2)) {
                result_files.push_back(ent->d_name);
                i++;
            }
        }
    }

    int choice;
    if (result_files.size()==0) {
        LOG(INFO)<<"No result files found! Please run calibration first.\n";
    } else if (result_files.size()==1) {
        temp_name = path_+result_dir_+"/"+result_files[0];
        file.open(temp_name.c_str());
    } else {
        LOG(INFO)<<"Multiple result files found!\n";
        for (int i=0; i<result_files.size(); i++) {
            LOG(INFO)<<i+1<<") "<<result_files[i]; // TODO: EXTRACT TIME AND DATE FROM FILENAME AND DISPLAY
            temp_name = path_+result_dir_+"/"+result_files[i];
            file.open(temp_name.c_str());
            string time_stamp;
            getline(file, time_stamp);
            LOG(INFO)<<" ("<<time_stamp<<")"<<endl;
            file.close();
        }
        LOG(INFO)<<"Select the file to load (1,..."<<result_files.size()<<"): ";
        cin>>choice;
        temp_name = path_+result_dir_+"/"+result_files[choice-1];
        file.open(temp_name.c_str());
    }

    string time_stamp;
    getline(file, time_stamp);

    double  reproj_error;
    file>>reproj_error;
    file>>img_size_.width;
    file>>img_size_.height;
    // file>>refocusing_params_.img_size.width;
    // file>>refocusing_params_.img_size.height;
    file>>pix_per_phys_;
    // file>>refocusing_params_.scale;
    file>>num_cams_;
    // file>>refocusing_params_.num_cams;

    //Mat_<double> P_u = Mat_<double>::zeros(3,4);
    Mat_<double> P = Mat_<double>::zeros(3,4);
    string cam_name;
    double tmp;
    
    for (int i=0; i<num_cams_; i++) {
        
        for (int j=0; j<2; j++) getline(file, cam_name);
        
        refocusing_params_.cam_names.push_back(cam_name);
        
        for (int j=0; j<3; j++) {
            for (int k=0; k<3; k++) {
                //file>>P_u(j,k);
            }
            //file>>P_u(j,3);
        }
        for (int j=0; j<3; j++) {
            for (int k=0; k<3; k++) {
                file>>P(j,k);
            }
            file>>P(j,3);
        }
        //refocusing_params_.P_mats_u.push_back(P_u.clone());
        refocusing_params_.P_mats.push_back(P.clone());

    }

    file>>refractive_;
    if(refractive_) {
        file>>geom[0];
        file>>geom[1];
        file>>geom[2];
        file>>geom[3];
        file>>geom[4];
    }

    file.close();

    LOG(INFO)<<"\nCALIBRATION RESULTS LOADED!\n";

}
*/

/* Probably not needed anymore
// Write BA results to files so Matlab can read them and plot world points and
// camera locations
void multiCamCalibration::write_calib_results_matlab() {

    ofstream file, file1, file2, file3, file4, file5;
    file.open("../matlab/world_points.txt");
    file1.open("../matlab/camera_points.txt");
    file2.open("../matlab/plane_params.txt");
    file3.open("../matlab/P_mats.txt");
    file4.open("../matlab/Rt.txt");
    file5.open("../matlab/f.txt");

    int* cameras = ba_problem_.camera_index();
    int* points = ba_problem_.point_index();
    int num_observations = ba_problem_.num_observations();
    const double* observations = ba_problem_.observations();

    int num_cameras = ba_problem_.num_cameras();
    double* camera_params = ba_problem_.mutable_cameras();

    int num_points = ba_problem_.num_points();
    double* world_points = ba_problem_.mutable_points();

    int num_planes = ba_problem_.num_planes();
    double* plane_params = ba_problem_.mutable_planes();

    for (int i=0; i<num_points; i++) {
        for (int j=0; j<3; j++) {
            file<<world_points[(i*3)+j]<<"\t";
        }
        file<<endl;
    }

    for (int i=0; i<num_planes; i++) {
        for (int j=0; j<4; j++) {
            file2<<plane_params[(i*4)+j]<<"\t";
        }
        file2<<endl;
    }

    for (int i=0; i<num_cameras; i++) {
        for (int j=0; j<3; j++) {
            for (int k=0; k<4; k++) {
                file3<<refocusing_params_.P_mats[i].at<double>(j,k)<<"\t";
            }
            file3<<endl;
        }
    }

    Mat_<double> rvec = Mat_<double>::zeros(1,3);
    Mat_<double> tvec = Mat_<double>::zeros(3,1);
    for (int i=0; i<num_cameras; i++) {
        
        file5<<camera_params[(i*9)+6]<<endl;

        for (int j=0; j<3; j++) {
            rvec.at<double>(0,j) = camera_params[(i*9)+j];
            tvec.at<double>(j,0) = camera_params[(i*9)+j+3];
        }
        
        Mat R;
        Rodrigues(rvec, R);

        Mat_<double> translation = Mat_<double>::zeros(1,3);
        translation = -R*tvec;
        for (int j=0; j<3; j++) {
            file1<<translation(j,0)<<"\t";
        }
        file1<<endl;
        
        for (int j=0; j<3; j++) {
            for (int k=0; k<3; k++) {
                file4<<R.at<double>(j,k)<<"\t";
            }
            file4<<tvec.at<double>(j,0)<<endl;
        }

    }

    file.close();
    file1.close();
    file2.close();
    file3.close();
    file4.close();
    file5.close();

}
*/

/* Probably not needed anymore
// Write BA results to files so Matlab can read them and plot world points and
// camera locations for refractive calibration
void multiCamCalibration::write_calib_results_matlab_ref() {

    ofstream file1, file2, file3, file4, file5;
    file1.open("../matlab/camera_points.txt");
    file2.open("../matlab/plane_params.txt");
    file3.open("../matlab/P_mats.txt");
    file4.open("../matlab/Rt.txt");
    file5.open("../matlab/f.txt");

    int* cameras = ba_problem_ref_.camera_index();
    int* points = ba_problem_ref_.point_index();
    int num_observations = ba_problem_ref_.num_observations();
    const double* observations = ba_problem_ref_.observations();

    int num_cameras = ba_problem_ref_.num_cameras();
    double* camera_params = ba_problem_ref_.mutable_cameras();

    int num_planes = ba_problem_ref_.num_planes();
    double* plane_params = ba_problem_ref_.mutable_planes();
    
    for (int i=0; i<num_planes; i++) {
        for (int j=0; j<6; j++) {
            file2<<plane_params[(i*6)+j]<<"\t";
        }
        file2<<endl;
    }

    for (int i=0; i<num_cameras; i++) {
        for (int j=0; j<3; j++) {
            for (int k=0; k<4; k++) {
                file3<<refocusing_params_.P_mats[i].at<double>(j,k)<<"\t";
            }
            file3<<endl;
        }
    }

    Mat_<double> rvec = Mat_<double>::zeros(1,3);
    Mat_<double> tvec = Mat_<double>::zeros(3,1);
    for (int i=0; i<num_cameras; i++) {
        
        file5<<camera_params[(i*9)+6]<<endl;

        for (int j=0; j<3; j++) {
            rvec.at<double>(0,j) = camera_params[(i*9)+j];
            tvec.at<double>(j,0) = camera_params[(i*9)+j+3];
        }
        
        Mat R;
        Rodrigues(rvec, R);

        Mat_<double> translation = Mat_<double>::zeros(1,3);
        translation = -R*tvec;
        for (int j=0; j<3; j++) {
            file1<<translation(j,0)<<"\t";
        }
        file1<<endl;
        
        for (int j=0; j<3; j++) {
            for (int k=0; k<3; k++) {
                file4<<R.at<double>(j,k)<<"\t";
            }
            file4<<tvec.at<double>(j,0)<<endl;
        }

    }

    file1.close();
    file2.close();
    file3.close();
    file4.close();
    file5.close();

}
*/

void multiCamCalibration::grid_view() {

    DIR *dir;
    struct dirent *ent;

    string dir1(".");
    string dir2("..");
    string temp_name;

    vector<string> img_names;

    dir = opendir(path_.c_str());
    while(ent = readdir(dir)) {
        temp_name = ent->d_name;
        if (temp_name.compare(dir1)) {
            if (temp_name.compare(dir2)) {
                if (temp_name.compare(temp_name.size()-3,3,"tif") == 0) {
                    string path_img = path_+temp_name;
                    img_names.push_back(path_img);
                }
            }
        }
    }

    vector<TIFF*> tiffs;
    for (int i=0; i<img_names.size(); i++) {
        TIFF* tiff = TIFFOpen(img_names[i].c_str(), "r");
        tiffs.push_back(tiff);
    }

    LOG(INFO)<<"Counting number of frames...";
    int dircount = 0;
    if (tiffs[0]) {
	do {
	    dircount++;
	} while (TIFFReadDirectory(tiffs[0]));
    }
    LOG(INFO)<<"done! "<<dircount<<" frames found."<<endl<<endl;

    Mat grid;
    int frame=0;
    int skip=50;
    while (frame<dircount) {
        
        for (int i=0; i<3; i++) {
            for (int j=0; j<3; j++) {

                Mat img;
                uint32 c, r;
                size_t npixels;
                uint32* raster;
            
                int ind = i*3+j;

                TIFFSetDirectory(tiffs[ind], frame);
                TIFFGetField(tiffs[ind], TIFFTAG_IMAGEWIDTH, &c);
                TIFFGetField(tiffs[ind], TIFFTAG_IMAGELENGTH, &r);
                npixels = r * c;
                raster = (uint32*) _TIFFmalloc(npixels * sizeof (uint32));
                if (raster != NULL) {
                    if (TIFFReadRGBAImageOriented(tiffs[ind], c, r, raster, ORIENTATION_TOPLEFT, 0)) {
                        img.create(r, c, CV_32F);
                        for (int row=0; row<r; row++) {
                            for (int col=0; col<c; col++) {
                                img.at<float>(row,col) = TIFFGetR(raster[row*c+col])/255.0;
                            }
                        }
                    }
                    _TIFFfree(raster);
                }
                
                grid.create(img.rows, img.cols, CV_32F);
                Mat small;
                resize(img, small, Size(img.cols/3,img.rows/3));
                small.copyTo(grid.colRange(i*grid.cols/3,i*grid.cols/3+grid.cols/3).rowRange(j*grid.rows/3,j*grid.rows/3+grid.rows/3));
                
            }
        }

        LOG(INFO)<<"frame "<<frame<<endl;
        imshow("frame", grid);
        if(waitKey(30) >= 0) break;

        frame += skip;

    }

}

// Function to get grid edge size in pixels
// Work: - write so that function uses a specific image that defines reference plane
void multiCamCalibration::get_grid_size_pix() {

    vector< vector<Point2f> > all_points;
    if (dummy_mode_) {
        all_points = all_corner_points_[center_cam_id_];
    } else {
        all_points = all_corner_points_[0];
    }

    vector<Point2f> points = all_points[0];

    double xdist=0;
    double ydist=0;
    double dist=0;
    int num=0;

    LOG(INFO)<<"\nCALCULATING GRID EDGE SIZE...\n";

    for (int i=1; i<grid_size_.width; i++) {
        for (int j=0; j<grid_size_.height; j++) {
            xdist = points[j*grid_size_.width+i].x-points[j*grid_size_.width+i-1].x;
            ydist = points[j*grid_size_.width+i].y-points[j*grid_size_.width+i-1].y;
            dist += sqrt(xdist*xdist+ydist*ydist);
            num++;
        }
    }

    /*
    for (int i=1; i<grid_size.height; i++) {
        for (int j=0; j<grid_size.width; j++) {
            xdist = points[j*grid_size.height+i].x-points[j*grid_size.height+i-1].x;
            ydist = points[j*grid_size.height+i].y-points[j*grid_size.height+i-1].y;
            //dist += sqrt(xdist*xdist+ydist*ydist);
            //LOG(INFO)<<sqrt(xdist*xdist+ydist*ydist)<<"\n";
            //num++;
        }
    }
    */

    grid_size_pix_ = dist/num;

    LOG(INFO)<<"GRID SIZE: "<<grid_size_pix_<<" pixels\n";

}

// Python wrapper
BOOST_PYTHON_MODULE(calibration) {

    using namespace boost::python;

    // class_<multiCamCalibration>("calibration", init<string, int, int, double, int, int, int, int , int>())
    //     .def("run", &multiCamCalibration::run)
    // ;

}
