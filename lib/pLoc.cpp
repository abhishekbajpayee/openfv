// -------------------------------------------------------
// -------------------------------------------------------
// Synthetic Aperture - Particle Tracking Velocimetry Code
// --- Particle Localization ---
// -------------------------------------------------------
// Author: Abhishek Bajpayee
//         Dept. of Mechanical Engineering
//         Massachusetts Institute of Technology
// -------------------------------------------------------
// -------------------------------------------------------

#include "std_include.h"
#include "refocusing.h"
#include "typedefs.h"
#include "pLoc.h"
#include "visualize.h"

using namespace std;
using namespace cv;

void pLocalize::run() {
    
    int frame = 0;
    find_particles_3d(frame);
    find_clusters();
    collapse_clusters();

}   

void pLocalize::find_particles_all_frames() {

    for (int i=0; i<refocus_.num_frames(); i++) {
        
        find_particles_3d(i);
        particles_all_.push_back(particles_);
        particles3D_.clear();
        clusters_.clear();
        particles_.clear();

    }

}

void pLocalize::find_particles_3d(int frame) {

    particle2d particle;
    vector<Point2f> points;
    vector<particle2d> particles;

    cout<<"\nSearching for particles through volume at frame "<<frame<<"...\n";

    for (float i=zmin_; i<=zmax_; i += dz_) {
        
        //cout<<int((i-zmin_)*100.0/(zmax_-zmin_));

        refocus_.GPUrefocus(i, thresh_, 0, frame);
        Mat image = refocus_.result;
        
        find_particles(image, points);
        refine_subpixel(image, points, particles);
        for (int j=0; j<particles.size(); j++) {
            particle.x = (particles[j].x - refocus_.img_size().width*0.5)/refocus_.scale();
            particle.y = (particles[j].y - refocus_.img_size().height*0.5)/refocus_.scale();
            particle.z = i;
            particle.I = particles[j].I;
            particles3D_.push_back(particle);
        }
        points.clear();
        particles.clear();        
        
        //cout<<"\r";

    }

    find_clusters();
    collapse_clusters();

    cout<<"\nSearch Complete! "<<particles_.size()<<" particles found.\n";

}

void pLocalize::z_resolution() {
    
    double zref = 5.0;
    double dz = 0.01;
    double bounds = 1.0;

    refocus_.GPUrefocus(zref, thresh_, 0, 0);
    Mat base = refocus_.result.clone();

    vector<double> x;
    vector<double> y;

    PyVisualize vis;

    for (float i=zref-bounds; i<=zref+bounds; i += dz) {
        
        refocus_.GPUrefocus(i, thresh_, 0, 0);
        Mat ref = refocus_.result.clone();

        Mat numMat = ref.mul(base);
        Scalar sumnum = sum(numMat);

        Mat denA = base.mul(base);
        Mat denB = ref.mul(ref);
        Scalar sumdenA = sum(denA);
        Scalar sumdenB = sum(denB);
        
        double num = sumnum.val[0];
        double den1 = sumdenA.val[0];
        double den2 = sumdenB.val[0];
        
        double cx = num/sqrt(den1*den2);

        x.push_back(i);
        y.push_back(cx);
        
    }
    
    vis.plot(x,y,"b");
    vis.show();

}

void pLocalize::crop_focus() {

    string imgpath("../cropped/thresh_150/");

    for (float i=-2.0; i<=2.0; i += 0.1) {

        refocus_.GPUrefocus(i, thresh_, 0, 0);
        Mat img = refocus_.result.clone();
        Mat cropped = img(Rect(647-5,480-5,10,10));
        char filename[10];
        sprintf(filename, "z_%.2f.jpg", i);
        string sfilename(filename);
        string fullpath = imgpath+sfilename;
        imwrite(fullpath, cropped);

    }

}
    
        

void pLocalize::find_clusters() {

    //cout<<"\nClustering found particles...";

    while(particles3D_.size()) {

        double x = particles3D_[0].x;
        double y = particles3D_[0].y;
        double z = particles3D_[0].z;

        double xydist, zdist;
        double xythresh = 0.5;
        double zthresh = 2.0;

        vector<int> used;

        vector<particle2d> single_particle;
        single_particle.push_back(particles3D_[0]);
        for (int i=1; i<particles3D_.size(); i++) {
            xydist = sqrt( pow((particles3D_[i].x-x), 2) + pow((particles3D_[i].y-y), 2) );
            zdist = abs( particles3D_[i].z -z );
            if (xydist<xythresh && zdist<zthresh) {
                single_particle.push_back(particles3D_[i]);
                used.push_back(i);
            }
        }
        
        // The tricky delete loop
        for (int i=used.size()-1; i>=0; i--) {
            particles3D_.erase(particles3D_.begin()+used[i]);
        }
        particles3D_.erase(particles3D_.begin());
        
        clusters_.push_back(single_particle);
        single_particle.clear();
        used.clear();

    }

    //cout<<"done!\n";
    //cout<<clusters_.size()<<" clusters found."<<endl;

    clean_clusters();

}

void pLocalize::clean_clusters() {

    //cout<<"\nCleaning clusters...\n";

    for (int i=clusters_.size()-1; i>=0; i--) {
        //cout<<clusters_[i].size()<<" ";
        if (clusters_[i].size() < cluster_size_) clusters_.erase(clusters_.begin()+i);
    }

    //cout<<"done!\n";

}

void pLocalize::collapse_clusters() {

    //cout<<"Collapsing clusters to 3D particles...";

    double xsum, ysum, zsum, den;
    Point3f point;

    for (int i=0; i<clusters_.size(); i++) {
        
        xsum = 0;
        ysum = 0;
        zsum = 0;
        den = clusters_[i].size();

        //cout<<i<<endl;
        for (int j=0; j<clusters_[i].size(); j++) {
            
            xsum += clusters_[i][j].x;
            ysum += clusters_[i][j].y;
            zsum += clusters_[i][j].z;
            //cout<<clusters_[i][j].z<<"\t"<<clusters_[i][j].I<<endl;

        }

        point.x = xsum/den;
        point.y = ysum/den;
        point.z = zsum/den;
        
        particles_.push_back(point);

    }

    //cout<<"done!\n";
    //cout<<particles_.size()<<" particles found."<<endl;

}

void pLocalize::find_particles(Mat image, vector<Point2f> &points_out) {

    int h = image.rows;
    int w = image.cols;
    int i_min = 0;
    int count_thresh = 6;
    Point2f tmp_loc;
    
    for (int i=0; i<h; i++) {
        for (int j=0; j<w; j++) {

            tmp_loc.x = j;
            tmp_loc.y = i;

            Scalar intensity = image.at<uchar>(i,j);
            int I = intensity.val[0];

            // Look for non zero value
            if (I>i_min && min_dist(tmp_loc, points_out)>2*window_) {
                
                Point2f l_max;
                l_max.x = j;
                l_max.y = i;
                double i_max = I;

                // Move to local peak
                for (int x=i-window_; x<=i+window_; x++) {
                    for (int y=j-window_; y<=j+window_; y++) {
                        
                        if (x<0 || x>=h || y<0 || y>=w) continue;

                        Scalar intensity2 = image.at<uchar>(x,y);
                        int I2 = intensity2.val[0];
                        if (I2>i_max) {
                            i_max = I2;
                            l_max.x = y;
                            l_max.y = x;
                        }
                        
                    }
                }
                
                // Find particle size in window
                int count=0;
                for (int x=l_max.y-1; x<=l_max.y+1; x++) {
                    for (int y=l_max.x-1; y<=l_max.x+1; y++) {
                        
                        if (x<0 || x>=h || y<0 || y>=w) continue;

                        Scalar intensity2 = image.at<uchar>(x,y);
                        int I2 = intensity2.val[0];
                        if (I2>i_min) count++;
                        
                    }
                }
                
                if (!point_in_list(l_max, points_out) && min_dist(l_max, points_out)>window_ && count>=count_thresh) {
                    points_out.push_back(l_max);
                }           

            }

        }
    }

}

void pLocalize::refine_subpixel(Mat image, vector<Point2f> points_in, vector<particle2d> &points_out) {
    
    int h = image.rows;
    int w = image.cols;

    double x_num, x_den, y_num, i_sum;
    int count;

    for (int i=0; i<points_in.size(); i++) {
        
        x_num=0;
        y_num=0;
        i_sum=0;
        count=0;

        for (int x=points_in[i].x-window_; x<=points_in[i].x+window_; x++) {
            for (int y=points_in[i].y-window_; y<=points_in[i].y+window_; y++) {

                if (x<0 || x>=w || y<0 || y>=h) continue;
                
                Scalar intensity = image.at<uchar>(y,x);
                int i_xy = intensity.val[0];
                x_num = x_num + double(i_xy)*(x+0.5);
                y_num = y_num + double(i_xy)*(y+0.5);
                i_sum = i_sum + double(i_xy);
                //if (i_xy>0) count++;

            }
        }

        particle2d point;
        point.x = x_num/i_sum;
        point.y = y_num/i_sum;
        point.I = i_sum/double(pow(window_+1,2));
        points_out.push_back(point);

    }

}

void pLocalize::write_all_particles_to_file(string path) {

    ofstream file;
    file.open(path.c_str());

    for (int i=0; i<particles_all_.size(); i++) {
        file<<particles_all_[i].size()<<endl;
        for (int j=0; j<particles_all_[i].size(); j++) {
            file<<particles_all_[i][j].x<<endl;
            file<<particles_all_[i][j].y<<endl;
            file<<particles_all_[i][j].z<<endl;
        }
    }

    file.close();

}

int pLocalize::point_in_list(Point2f point, vector<Point2f> points) {

    int result=0;

    for (int i=0; i<points.size(); i++) {
        if (points[i].x==point.x && points[i].y==point.y) {
            result=1;
            break;
        }
    }

    return(result);

}

double pLocalize::min_dist(Point2f point, vector<Point2f> points) {

    double m_dist = 100.0;
    double dist;

    for (int i=0; i<points.size(); i++) {
        dist = sqrt( pow(points[i].x-point.x, 2)+ pow(points[i].y-point.y, 2) );
        if (dist<m_dist) m_dist = dist;
    }

    return(m_dist);

}

void pLocalize::draw_points(Mat image, Mat &drawn, vector<Point2f> points) {

    cvtColor(image, drawn, CV_GRAY2RGB);

    for (int i=0; i<points.size(); i++) {
        
    }

    for (int i=0; i<points.size(); i++) {
        //points[i].x = points[i].x*10;
        //points[i].y = points[i].y*10;
        circle(drawn, points[i], 5.0, Scalar(0,0,255));
    }

}

void pLocalize::draw_point(Mat image, Mat &drawn, Point2f point) {

    cvtColor(image, drawn, CV_GRAY2RGB);
    circle(drawn, point, 5.0, Scalar(0,0,255));

}
