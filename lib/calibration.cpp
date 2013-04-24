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

void multiCamCalibration::initialize() {

    ba_file_ = string("temp/ba_data.txt");
    result_dir_ = string("calibration_results");
    
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
            cout<<"\'"<<result_dir_<<"\' directory found in "<<path_<<"! Calibration has already been performed earlier. Please select what to do...\n1) Run calibration again\n2) Read results\nEnter your choice (1/2): ";
            cin>>choice;
            if (choice==1) {
                run_calib_flag = 1;
            } else if (choice==2) {
                load_results_flag = 1;
            } else {
                cout<<"Invalid choice!\n";
            }
        }
    }
    if (!result_dir_found) run_calib_flag = 1;

    center_cam_id_ = 4; // TODO

    show_corners_flag = 0;
    results_just_saved_flag = 0;

}

void multiCamCalibration::run() {
    
    initialize();

    if (run_calib_flag) {
        read_cam_names();
        read_calib_imgs();
        find_corners();
        initialize_cams();
        write_BA_data();
        run_BA();
        write_calib_results();
    }

    if (load_results_flag) {
        load_calib_results();
    }

    cout<<"\nCALIBRATION COMPLETE!\n";

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

    cout<<"\nREADING CAMERA NAMES...\n\n";
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
    
    for (int i=0; i<cam_names_.size(); i++) cout<<"Camera "<<i+1<<": "<<cam_names_[i]<<endl;

    num_cams_ = cam_names_.size();

}

void multiCamCalibration::read_calib_imgs() {

    DIR *dir;
    struct dirent *ent;
 
    string dir1(".");
    string dir2("..");
    string temp_name;
    string img_prefix = "";

    vector<string> img_names;
    Mat image;

    cout<<"\nREADING IMAGES...\n\n";

    for (int i=0; i<num_cams_; i++) {

        cout<<"Camera "<<i+1<<" of "<<num_cams_<<"...";

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
        for (int i=0; i<img_names.size(); i++) {
            image = imread(img_names[i], 0);
            calib_imgs_sub.push_back(image);
        }
        img_names.clear();

        calib_imgs_.push_back(calib_imgs_sub);
        path_tmp = "";

        cout<<"done!\n";
   
    }
 
    num_imgs_ = calib_imgs_[0].size();
    img_size_ = Size(calib_imgs_[0][0].cols, calib_imgs_[0][0].rows);
    refocusing_params.img_size = img_size_;

    cout<<"\nDONE READING IMAGES!\n\n";

}

// TODO: add square grid correction capability again
void multiCamCalibration::find_corners() {

    vector< vector<Point2f> > corner_points;
    vector<Point2f> points;
    Mat scene, scene_gray, scene_drawn;
    
    cout<<"\nFINDING CORNERS...\n\n";
    for (int i=0; i<num_cams_; i++) {
        
        cout<<"Camera "<<i+1<<" of "<<num_cams_;

        vector<Mat> imgs_temp = calib_imgs_[i];
        for (int j=0; j<imgs_temp.size(); j++) {

            //equalizeHist(imgs_temp[j], scene);
            scene = imgs_temp[j];
            bool found = findChessboardCorners(scene, grid_size_, points, CV_CALIB_CB_ADAPTIVE_THRESH);
            
            if (found) {
                //cvtColor(scene, scene_gray, CV_RGB2GRAY);
                cornerSubPix(scene, points, Size(20, 20), Size(-1, -1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
                corner_points.push_back(points);
                
                if (show_corners_flag) {
                    scene_drawn = scene;
                    drawChessboardCorners(scene_drawn, grid_size_, points, found);
                    namedWindow("Pattern", CV_WINDOW_AUTOSIZE);
                    imshow("Pattern", scene);
                    waitKey(0);
                    cvDestroyWindow("Pattern");
                }
                
            } else {
                cout<<"pattern not found!\n";
            }

        }

        all_corner_points_.push_back(corner_points);
        corner_points.clear();
        cout<<"\n";

    }

    get_grid_size_pix();
    pix_per_phys_ = grid_size_pix_/grid_size_phys_;
    refocusing_params.scale = pix_per_phys_;

    cout<<"\nCORNER FINDING COMPLETE!\n\n";

}

void multiCamCalibration::initialize_cams() {

    vector<Point3f> pattern_points;
    float xshift = -(grid_size_.width-1)/2.0;
    float yshift = -(grid_size_.height-1)/2.0;
    for (int i=0; i<grid_size_.height; i++) {
        for (int j=0; j<grid_size_.width; j++) {
            pattern_points.push_back(Point3f( 5*(float(j)+xshift), 5*(float(i)+yshift), 0.0f));
        }
    }
    for (int i=0; i<num_imgs_; i++) {
        all_pattern_points_.push_back(pattern_points);
    }

    cout<<"\nINITIALIZING CAMERAS...\n\n";

    for (int i=0; i<num_cams_; i++) {

        cout<<"Calibrating camera "<<i+1<<"...";
        Mat_<double> A = Mat_<double>::zeros(3,3); 
        Mat_<double> dist_coeff;
        
        A(0,0) = 9000; 
        A(1,1) = 9000;
        A(0,2) = 646; 
        A(1,2) = 482;
        A(2,2) = 1;
        
        vector<Mat> rvec, tvec;
        calibrateCamera(all_pattern_points_, all_corner_points_[i], img_size_, A, dist_coeff, rvec, tvec, CV_CALIB_USE_INTRINSIC_GUESS|CV_CALIB_FIX_PRINCIPAL_POINT|CV_CALIB_FIX_ASPECT_RATIO);
        //cout<<calibrateCamera(all_pattern_points, all_corner_points[i], img_size, A, dist_coeff, rvec, tvec, CV_CALIB_FIX_ASPECT_RATIO)<<endl;
        cout<<"done!\n";

        cameraMats_.push_back(A);
        dist_coeffs_.push_back(dist_coeff);
        rvecs_.push_back(rvec);
        tvecs_.push_back(tvec);

    }

    cout<<"\nCAMERA INITIALIZATION COMPLETE!\n\n";

}

void multiCamCalibration::write_BA_data() {

    cout<<"\nWRITING POINT DATA TO FILE...";
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
        
        file<<cameraMats_[i].at<double>(0,0)<<"\t";
        
        //for (int j=0; j<2; j++) file<<dist_coeffs[i].at<double>(0,j)<<"\n";
        file<<0<<"\t"<<0<<endl;
    }
    
    double width = 25;
    double height = 20;
    // add back projected point guesses here
    for (int i=0; i<num_points; i++) {
        file<<(double(rand()%50)-(width*0.5))<<"\t";
        file<<(double(rand()%50)-(height*0.5))<<"\t";
        file<<(rand()%50)<<"\t";
        file<<endl;
    }

    /*
    for (int k=0; k<9; k++) {
    for (int i=0; i<6; i++) {
        for (int j=0; j<5; j++) {
            file<<(5*i)-25<<endl<<(5*j)-20<<endl<<(5*k)-20<<endl;
            //file2<<(5*i)-25<<endl<<(5*j)-20<<endl<<(5*k)-20<<endl;
        }
    }
    }
    */

    param = 1;
    for (int i=0; i<imgs_per_cam; i++) {
        for (int j=0; j<4; j++) {
            file<<param<<"\t";
        }
        file<<endl;
    }

    file.close();
    cout<<"DONE!\n";

}

void multiCamCalibration::run_BA() {

    total_reproj_error_ = BA_pinhole(ba_problem_, ba_file_, img_size_);
    avg_reproj_error_ = total_reproj_error_/double(ba_problem_.num_observations());
    cout<<"FINAL TOTAL REPROJECTION ERROR: "<<total_reproj_error_<<endl;

}

void multiCamCalibration::write_calib_results() {

    char choice;
    cout<<"\nDo you wish to save the calibration results (y/n)?: ";
    cin>>choice;

    if (choice==89 || choice==121) {
        
        string newPath = path_ + result_dir_ + "/";
        if (!dirExists(newPath)) {
            cout<<"\'calibration_results\' directory does not exist..."<<endl;
            mkdir(newPath.c_str(), S_IRWXU);
            cout<<"directory created!"<<endl;
        }

        time_t timer;
        char time_stamp[15];
        sprintf(time_stamp, "%.15ld", timer);
        string time_stamp_str(time_stamp);
        result_file_ = newPath + "results_"+time_stamp_str+".txt";
        results_just_saved_flag = 1;
        
        struct tm * timeinfo;
        time (&timer);
        timeinfo = localtime(&timer);
        char time_stamp_hr[50];
        sprintf(time_stamp_hr, "Calibration performed: %s", asctime(timeinfo));
        string time_stamp_hr_str(time_stamp_hr);

        double* camera_params = ba_problem_.mutable_cameras();

        ofstream file;
        file.open(result_file_.c_str());

        // WRITING DATA TO RESULTS FILE
        
        file<<time_stamp_hr_str<<endl<<endl;
        file<<total_reproj_error_<<endl<<avg_reproj_error_<<endl<<endl;
        file<<num_cams_<<endl<<endl;

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
            
            file<<cam_names_[i]<<endl;
            refocusing_params.cam_names.push_back(cam_names_[i]);
            file<<camera_params[(i*num_cams_)+6]<<"\t";
            file<<camera_params[(i*num_cams_)+7]<<"\t";
            file<<camera_params[(i*num_cams_)+8]<<"\t"<<endl;
            for (int j=0; j<3; j++) {
                for (int k=0; k<3; k++) {
                    file<<R.at<double>(j,k)<<"\t";
                }
                file<<tvec(j,0)<<endl;
            }
            file<<endl;
            
            K(0,0) = camera_params[(i*num_cams_)+6];
            K(1,1) = camera_params[(i*num_cams_)+6];
            K(0,2) = img_size_.width*0.5;
            K(1,2) = img_size_.height*0.5;
            K(2,2) = 1;

            rVecs_.push_back(rvec.clone());
            tVecs_.push_back(tvec.clone());
            K_mats_.push_back(K.clone());

        }
        
        Mat_<double> P_u = Mat_<double>::zeros(3,4);
        Mat_<double> P = Mat_<double>::zeros(3,4);
        Mat_<double> rmean = Mat_<double>::zeros(3,3);
        matrixMean(rVecs_, rmean);
        cout<<rmean<<endl;/*
        for (int i=0; i<num_cams_; i++) {
            P_from_KRT(K_mats_[i], rVecs_[i], tVecs_[i], rmean, P_u, P);
            refocusing_params.P_mats_u.push_back(P_u.clone());
            refocusing_params.P_mats.push_back(P.clone());
            }*/
        
        file<<img_size_.width<<"\t"<<img_size_.height<<"\t"<<pix_per_phys_;

        file.close();

        cout<<"\nCalibration results saved to file: "<<result_file_<<endl;

    }

}

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

    cout<<"\nLOOKING FOR CALIBRATION RESULT FILES...\n\n";
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
        cout<<"No result files found! Please run calibration first.\n";
    } else if (result_files.size()==1) {
        file.open(result_files[0].c_str());
    } else {
        cout<<"Multiple result files found!\n";
        for (int i=0; i<result_files.size(); i++) {
            cout<<i+1<<") "<<result_files[i]<<endl;
        }
        cout<<"Select the file to laod (1,..."<<result_files.size()<<"): ";
        cin>>choice;
        file.open(result_files[choice-1].c_str());
    }

    file.close();


}

// Function to get grid edge size in pixels
// Work: - write so that function uses a specific image that defines reference plane
void multiCamCalibration::get_grid_size_pix() {

    vector< vector<Point2f> > all_points = all_corner_points_[center_cam_id_];
    vector<Point2f> points = all_points[0];

    double xdist=0;
    double ydist=0;
    double dist=0;
    int num=0;

    cout<<"\nCALCULATING GRID EDGE SIZE...\n";

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
            //cout<<sqrt(xdist*xdist+ydist*ydist)<<"\n";
            //num++;
        }
    }
    */

    grid_size_pix_ = dist/num;

    cout<<"GRID SIZE: "<<grid_size_pix_<<" pixels\n";

}
