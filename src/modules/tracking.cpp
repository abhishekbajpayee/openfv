//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//                           License Agreement
//                For Open Source Flow Visualization Library
//
// Copyright 2013-2017 Abhishek Bajpayee
//
// This file is part of OpenFV.
//
// OpenFV is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License version 2 as published by the Free Software Foundation.
//
// OpenFV is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License version 2 for more details.
//
// You should have received a copy of the GNU General Public License version 2 along with
// OpenFV. If not, see https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html.

// -------------------------------------------------------
// -------------------------------------------------------
// Synthetic Aperture - Particle Tracking Velocimetry Code
// --- Particle Tracking ---
// -------------------------------------------------------
// Author: Abhishek Bajpayee
//         Dept. of Mechanical Engineering
//         Massachusetts Institute of Technology
// -------------------------------------------------------
// -------------------------------------------------------

#include "tracking.h"

using namespace std;
using namespace cv;

pLocalize::pLocalize(localizer_settings s, saRefocus refocus, refocus_settings s2):
    window_(s.window), zmin_(s.zmin), zmax_(s.zmax), dz_(s.dz), thresh_(s.thresh), zmethod_(s.zmethod), refocus_(refocus), s2_(s2), show_particles_(s.show_particles), show_refocused_(s.show_refocused), cluster_size_(s.cluster_size) {

    zext_ = 2.5;

    //cout<<"Crit cluster size: "<<cluster_size_<<endl;

}

void pLocalize::run() {

    int frame = 0;
    find_particles_3d(frame);

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
    vector<Point2f> points, points2;
    vector<particle2d> particles;

    double rx = 0; double ry = 0; double rz = 0;

    VLOG(2)<<"Searching for particles through volume at frame "<<frame<<"..."<<endl;

    for (float i=zmin_; i<=zmax_; i += dz_) {

        VLOG(3)<<1+int((i-zmin_)*100.0/(zmax_-zmin_))<<"%"<<flush;

        Mat image = refocus_.refocus(i, rx, ry, rz, thresh_, frame);
        if (show_refocused_) {
            qimshow(image);
        }

        find_particles(image, points);

        refine_subpixel(image, points, particles);
        if (show_particles_) {
            Mat img; draw_points(image, img, particles); pimshow(img, i, particles.size()); cout<<"\r"<<i<<flush;
        }

        for (int j=0; j<particles.size(); j++) {
            particle.x = (particles[j].x - refocus_.img_size().width*0.5)/refocus_.scale();
            particle.y = (particles[j].y - refocus_.img_size().height*0.5)/refocus_.scale();
            particle.z = i;
            particle.I = particles[j].I;
            particles3D_.push_back(particle);
        }
        points.clear();
        particles.clear();

    }

    int write_clust=1;
    if (write_clust)
        write_clusters(particles3D_, "../temp/clusters.txt");

    find_clusters();
    collapse_clusters();

    cout<<"\rdone! "<<particles_.size()<<" particles found.\n";

}

void pLocalize::z_resolution() {

    double zref = 5.0;
    double dz = 1.0;
    double bounds = 1.0;


    ofstream file;
    file.open("../../cx_v_dz_t100.txt");

    double t = 40.0;

    double factor = 0.5;
    for (int i=0; i<10; i++) {

        Mat base = refocus_.refocus(zref, 0, 0, 0, thresh_, 0);

        Mat ref = refocus_.refocus(zref+dz, 0, 0, 0, thresh_, 0);

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

        file<<dz<<"\t"<<cx<<endl;

        dz *= factor;

    }

    file.close();

}

void pLocalize::find_clusters() {

    VLOG(1)<<"\nClustering found particles...";

    while(particles3D_.size()) {

        double x = particles3D_[0].x;
        double y = particles3D_[0].y;
        double z = particles3D_[0].z;

        double xydist, zdist;
        double xythresh = 2.0/refocus_.scale();
        double zthresh = zext_;

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

    VLOG(1)<<"done!\n";
    VLOG(1)<<clusters_.size()<<" clusters found."<<endl;

    clean_clusters();

}

void pLocalize::clean_clusters() {

    VLOG(1)<<"Cleaning clusters...\n";

    for (int i=clusters_.size()-1; i>=0; i--) {
        //cout<<clusters_[i].size()<<" ";
        if (clusters_[i].size() < cluster_size_) clusters_.erase(clusters_.begin()+i);
    }

    VLOG(1)<<"done!\n";

}

void pLocalize::collapse_clusters() {

    //cout<<"Collapsing clusters to 3D particles...";

    double xsum, ysum, zsum, den;
    Point3f point;

    for (int i=0; i<clusters_.size(); i++) {

        xsum = 0;
        ysum = 0;
        den = clusters_[i].size();

        for (int j=0; j<clusters_[i].size(); j++) {

            xsum += clusters_[i][j].x;
            ysum += clusters_[i][j].y;

        }

        point.x = xsum/den;
        point.y = ysum/den;

        point.z = get_zloc(clusters_[i]);

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

            // Scalar intensity = image.at<uchar>(i,j);
            // int I = intensity.val[0];

            float I = image.at<float>(i,j);

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

                        // Scalar intensity2 = image.at<uchar>(x,y);
                        // int I2 = intensity2.val[0];

                        float I2 = image.at<float>(x,y);

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

                        // Scalar intensity2 = image.at<uchar>(x,y);
                        // int I2 = intensity2.val[0];

                        float I2 = image.at<float>(x,y);

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

                // Scalar intensity = image.at<uchar>(y,x);
                // int i_xy = intensity.val[0];

                float i_xy = image.at<float>(y,x);

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

    file<<particles_all_.size()<<endl;

    for (int i=0; i<particles_all_.size(); i++) {
        file<<particles_all_[i].size()<<endl;
        for (int j=0; j<particles_all_[i].size(); j++) {
            file<<particles_all_[i][j].x<<"\t";
            file<<particles_all_[i][j].y<<"\t";
            file<<particles_all_[i][j].z<<endl;
        }
    }

    file.close();

    cout<<"All particles written to: "<<path<<endl;

}

void pLocalize::write_all_particles(string path) {

    stringstream s;
    s<<path<<"particles/";
    s<<"f"<<s2_.start_frame<<"to"<<s2_.end_frame<<"_";
    s<<"w"<<window_<<"_";
    s<<"t"<<thresh_<<"_";
    s<<"zm"<<zmethod_<<"_";
    s<<"idz"<<1/dz_<<".txt";
    string particle_file = s.str();

    ofstream file;
    file.open(particle_file.c_str());

    file<<particles_all_.size()<<endl;

    for (int i=0; i<particles_all_.size(); i++) {
        file<<particles_all_[i].size()<<endl;
        for (int j=0; j<particles_all_[i].size(); j++) {
            file<<particles_all_[i][j].x<<"\t";
            file<<particles_all_[i][j].y<<"\t";
            file<<particles_all_[i][j].z<<endl;
        }
    }

    file.close();

    cout<<"All particles written to: "<<particle_file<<endl;

}

void pLocalize::write_particles_to_file(string path) {

    ofstream file;
    file.open(path.c_str());

    for (int i=0; i<particles_.size(); i++) {
        file<<particles_[i].x<<"\t";
        file<<particles_[i].y<<"\t";
        file<<particles_[i].z<<endl;
        cout<<i<<endl;
    }

    file.close();

}

void pLocalize::write_clusters(vector<particle2d> &particles3D_, string path) {

    VLOG(1)<<"Writing clusters to file...";

    ofstream file;
    file.open(path.c_str());
    for (int i=0; i<particles3D_.size(); i++) {
        file<<particles3D_[i].x<<"\t";
        file<<particles3D_[i].y<<"\t";
        file<<particles3D_[i].z<<"\t";
        file<<particles3D_[i].I<<"\n";
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
        circle(drawn, points[i], 5.0, Scalar(0,0,255));
    }

}

void pLocalize::draw_points(Mat image, Mat &drawn, vector<particle2d> points) {

    cvtColor(image, drawn, CV_GRAY2RGB);

    for (int i=0; i<points.size(); i++) {
        circle(drawn, Point2f(points[i].x, points[i].y), 5.0, Scalar(0,0,255));
    }

}

void pLocalize::draw_point(Mat image, Mat &drawn, Point2f point) {

    cvtColor(image, drawn, CV_GRAY2RGB);
    circle(drawn, point, 5.0, Scalar(0,0,255));

}

double pLocalize::get_zloc(vector<particle2d> cluster) {

    double z;

    switch (zmethod_) {

    case 1: { // mean method

        double den = cluster.size();
        double zsum = 0;
        for (int i=0; i<cluster.size(); i++)
            zsum += cluster[i].z;

        z = zsum/den;
        break;

    }

    case 2: { // poly2 fit

        double* params = new double[3];
        ceres::Problem problem;
        for (int i=0; i<cluster.size(); i++) {

            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<poly2FitError, 1, 3>
                (new poly2FitError(cluster[i].z, cluster[i].I));

            problem.AddResidualBlock(cost_function,
                                     NULL,
                                     params);

        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        z = -(params[1]/(2*params[0]));
        break;

    }

    case 3: { // gauss fit

        double* params = new double[3];

        // initial values for parameters
        params[1] = cluster[0].z; params[0] = 1; params[2] = 0.25;

        ceres::Problem problem;
        for (int i=0; i<cluster.size(); i++) {

            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<gaussFitError, 1, 3>
                (new gaussFitError(cluster[i].z, cluster[i].I));

            problem.AddResidualBlock(cost_function,
                                     NULL,
                                     params);

        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        z = params[1];
        break;

    }

    }

    return(z);

}

pTracking::pTracking(string particle_file, double Rn, double Rs) {

    path_ = particle_file; R_s = Rs; R_n = Rn;

    initialize();
    read_points();

}

void pTracking::initialize() {

    N = 5000;
    tol = 0.01;

    V_n = (4/3)*pi*pow(R_n,3);
    V_s = (4/3)*pi*pow(R_s,3);

    A = 0.3;
    B = 3.0;
    C = 0.1;
    D = 5.0;
    E = 1.0;
    F = 0.05;

    method_ = 1;

}

void pTracking::set_vars(int method, double rn, double rs, double e, double f) {

    method_ = method;
    // reject_singles_ = reject_singles;

    R_n = rn;
    R_s = rs;

    E = e;
    F = f;

    match_counts.clear();
    all_matches.clear();

}

void pTracking::read_points() {

    Point3f point;
    volume vol;

    ifstream file;
    file.open(path_.c_str());

    VLOG(1)<<"Reading points to track...";

    int num_frames;
    file>>num_frames;

    for (int i=0; i<num_frames; i++) {

        int num_points;
        file>>num_points;
        VLOG(1)<<"f"<<i<<": "<<num_points<<", ";

        vol.x1 = 0;
        vol.x2 = 0;
        vol.y1 = 0;
        vol.y2 = 0;
        vol.z1 = 0;
        vol.z2 = 0;

        for (int j=0; j<num_points; j++) {

            file>>point.x;
            file>>point.y;
            file>>point.z;

            if (point.x>vol.x2) vol.x2 = point.x;
            if (point.y>vol.y2) vol.y2 = point.y;
            if (point.z>vol.z2) vol.z2 = point.z;
            if (point.x<vol.x1) vol.x1 = point.x;
            if (point.y<vol.y1) vol.y1 = point.y;
            if (point.z<vol.z1) vol.z1 = point.z;

            points_.push_back(point);
        }

        vol.v = (vol.x2-vol.x1)*(vol.y2-vol.y1)*(vol.z2-vol.z1);
        double density = points_.size()/vol.v;
        double complexity = 300*points_.size()*pow(density, 4)*pow(R_s*R_n,6);
        //cout<<"complexity: "<<complexity<<endl;
        all_points_.push_back(points_);
        vols_.push_back(vol);
        points_.clear();

    }

    VLOG(1)<<"done!"<<endl;

}

void pTracking::track_frames(int start, int end) {

    offset = start;

    vector<Point2i> matches;

    for (int i=start; i<end; i++) {

        int count;
        matches = track_frame(i, i+1, count);
        all_matches.push_back(matches);
        match_counts.push_back(count);
        matches.clear();

    }


}

void pTracking::track_all() {

    offset = 0;

    vector<Point2i> matches;

    for (int i=0; i<all_points_.size()-1; i++) {
        //for (int i=0; i<5; i++) {

        int count;
        matches = track_frame(i, i+1, count);
        all_matches.push_back(matches);
        match_counts.push_back(count);
        matches.clear();

    }

}

vector<Point2i> pTracking::track_frame(int f1, int f2, int &count) {

    LOG(INFO)<<"Matching frames "<<f1<<" and "<<f2<<" | ";

    int n1, n2;
    n1 = all_points_[f1].size();
    n2 = all_points_[f2].size();

    VLOG(1)<<"Neighbor Sets...";
    // vector< vector<int> > S_r = neighbor_set(f1, f1, R_n);
    // vector< vector<int> > S_c = neighbor_set(f1, f2, R_s);

    vector< vector<int> > S_r = neighbor_set(f1, R_n);
    vector< vector<int> > S_c = candidate_set(f1, f2, R_s);

    vector<Mat> Pij, Pi, Pij2, Pi2;
    VLOG(1)<<"Probability Sets...";
    build_probability_sets(S_r, S_c, Pij, Pi, Pij2, Pi2);

    VLOG(1)<<"Relaxation Sets...\n";
    vector< vector< vector< vector<Point2i> > > > theta;
    //double t = omp_get_wtime();
    build_relaxation_sets(f1, f2, S_r, S_c, C, D, E, F, theta);
    //cout<<"Time: "<<omp_get_wtime()-t<<endl;

    double diff;
    int n;
    for (n=0; n<N; n++) {

        for (int k=0; k<Pij.size(); k++) {

            for (int i=0; i<Pij[k].rows; i++) {
                for (int j=0; j<Pij[k].cols; j++) {

                    double sum = 0;
                    for (int l=0; l<theta[k][i][j].size(); l++) {
                        sum += Pij[k].at<double>(theta[k][i][j][l].x, theta[k][i][j][l].y);
                    }

                    double newval = (Pij[k].at<double>(i,j))*(A + (B*sum));
                    Pij2[k].at<double>(i,j) = newval;

                }
            }

        }

        normalize_probabilites(Pij2,Pi2);
        diff = update_probabilities(Pij, Pi, Pij2, Pi2);

        if (diff<tol) break;

    }

    VLOG(1)<<"Final residual change: "<<diff<<" in "<<n+1<<" iterations. "<<endl;

    vector<Point2i> matches;
    count = find_matches(Pij, Pi, S_r, S_c, matches);

    return(matches);

}

void pTracking::track_frame_n(int f1, int f2) {

    LOG(INFO)<<"Matching frames "<<f1<<" and "<<f2<<" | ";

    int n1, n2;
    n1 = all_points_[f1].size();
    n2 = all_points_[f2].size();

    VLOG(1)<<"Neighbor Sets...";
    vector< vector<int> > S_r = neighbor_set(f1, R_n);
    vector< vector<int> > S_c = candidate_set(f1, f2, R_s);

    Mat_<double> Pij = Mat_<double>::zeros(n1, n2); Mat_<double> Pij2 = Mat_<double>::zeros(n1, n2);
    Mat_<double> Pi = Mat_<double>::zeros(n1, 1); Mat_<double> Pi2 = Mat_<double>::zeros(n1, 1);
    VLOG(1)<<"Probability Sets...";
    build_probability_sets_n(S_r, S_c, Pij, Pi, Pij2, Pi2);

    VLOG(1)<<"Relaxation Sets...\n";
    vector< vector< vector<Point2i> > > theta;
    //double t = omp_get_wtime();
    build_relaxation_sets_n(f1, f2, S_r, S_c, C, D, E, F, theta);
    //cout<<"Time: "<<omp_get_wtime()-t<<endl;

    double diff;
    int n;

    for (n=0; n<N; n++) {

        //for (int i=0; i<Pij.size(); i++) {
        for (int i=0; i<S_r.size(); i++) {

            //for (int j=0; j<Pij[i].size(); j++) {
            for (int j=0; j<S_c[i].size(); j++) {

                double sum = 0;
                for (int k=0; k<theta[i][j].size(); k++) {
                    //VLOG(3)<<theta[i][j].size();
                    sum += Pij(theta[i][j][k].x,theta[i][j][k].y);
                }

                double newval = Pij(i,S_c[i][j])*( A + (B*sum) );
                Pij2(i,S_c[i][j]) = newval;

            }

        }

        normalize_probabilites_n(Pij2,Pi2);

        diff = double(sum(abs(Pij-Pij2))[0] + sum(abs(Pi-Pi2))[0]);
        VLOG(3)<<n+1<<": "<<diff;
        Pij = Pij2.clone();
        Pi = Pi2.clone();

        if (diff<tol) break;

    }

    VLOG(1)<<"Final residual change: "<<diff<<" in "<<n+1<<" iterations. "<<endl;

    vector<Point2i> matches;
    // count = find_matches_n(Pij, Pi, matches);

    P_ = Pij;

    //return(matches);

}

int pTracking::find_matches(vector<Mat> Pij, vector<Mat> Pi, vector< vector<int> > S_r, vector< vector<int> > S_c, vector<Point2i> &matches) {

    vector< vector<int> > zematch;
    vector<int> container;
    for (int i=0; i<Pij.size(); i++)
        zematch.push_back(container);

    for (int i=0; i<Pij.size(); i++) {

        cout<<Pij[i]<<endl;
        for (int j=0; j<Pij[i].rows; j++) {
            for (int k=0; k<Pij[i].cols; k++) {

                if (Pij[i].at<double>(j,k)>0.99) {
                    zematch[S_r[i][j]].push_back(S_c[i][k]);
                }

            }
        }

    }

    int count = 0;
    int count_single = 0;
    int count_mult = 0;
    int tiecount = 0;
    for (int i=0; i<Pij.size(); i++) {

        vector<int> c;
        vector<int> cf;

        if (zematch[i].size()==0) {
            matches.push_back(Point2i(i,-1));
        } else if (zematch[i].size()==1) {
            if (!reject_singles_) {
                matches.push_back(Point2i(i,zematch[i][0]));
                count++;
                count_single++;
            }
        } else {

            for (int j=0; j<zematch[i].size(); j++) {

                int counted=0;
                for (int k=0; k<c.size(); k++) {
                    if (zematch[i][j]==c[k]) {
                        counted=1;
                        cf[k]++;
                    }
                }
                if (counted==0) {
                    c.push_back(zematch[i][j]);
                    cf.push_back(1);
                }

            }

            // Finding maximum occurence frequency
            int modloc = 0;
            int modf = cf[0];
            for (int k=1; k<cf.size(); k++) {
                if (cf[k]>modf) {
                    modloc = k;
                    modf = cf[k];
                }
            }

            // Checking if multiple maxima exist
            int tie = 0;
            for (int k=0; k<cf.size(); k++) {

                if (k==modloc)
                    continue;

                if (cf[k]==cf[modloc]) {
                    tie=1;
                    tiecount++;
                    matches.push_back(Point2i(i,-2));
                    break;
                }

            }

            if (tie==0) {
                matches.push_back(Point2i(i,c[modloc]));
                count++;
                count_mult++;
            }

        }

        c.clear(); cf.clear();

    }

    LOG(INFO)<<"Ties: "<<tiecount<<", Matches: "<<count<<", Ratio: "<<double(count)/double(Pij.size())<<endl;
    LOG(INFO)<<"Singles: "<<count_single<<", Multiples: "<<count_mult;

    return(count);

}

double pTracking::update_probabilities(vector<Mat> &Pij, vector<Mat> &Pi, vector<Mat> &Pij2, vector<Mat> &Pi2) {

    Mat diffpij, diffpi;
    double difftotal = 0;
    Scalar diffsum;

    for (int i=0; i<Pij.size(); i++) {

        if (Pij[i].rows>0 && Pij[i].cols>0) {
            absdiff(Pij[i], Pij2[i], diffpij);
            diffsum = sum(diffpij);
            difftotal += diffsum.val[0];
        }

        Pij[i] = Pij2[i].clone();

        if (Pi[i].rows>0 && Pi[i].cols>0) {
            absdiff(Pi[i], Pi2[i], diffpi);
            diffsum = sum(diffpi);
            difftotal += diffsum.val[0];
        }

        Pi[i] = Pi2[i].clone();
    }

    return(difftotal);

}

void pTracking::normalize_probabilites(vector<Mat> &Pij, vector<Mat> &Pi) {

    for (int i=0; i<Pij.size(); i++) {

        for (int j=0; j<Pij[i].rows; j++) {
            double sum = 0;
            for (int k=0; k<Pij[i].cols; k++) {
                sum += Pij[i].at<double>(j,k);
            }
            sum += Pi[i].at<double>(j,0);
            for (int k=0; k<Pij[i].cols; k++) {
                Pij[i].at<double>(j,k) /= sum;
            }
            Pi[i].at<double>(j,0) /= sum;
        }

    }

}

void pTracking::build_probability_sets(vector< vector<int> > S_r, vector< vector<int> > S_c, vector<Mat> &Pij, vector<Mat> &Pi, vector<Mat> &Pij2, vector<Mat> &Pi2) {

    for (int i=0; i<S_r.size(); i++) {

        Mat_<double> Pij_single = Mat_<double>::zeros(S_r[i].size(), S_c[i].size());
        Mat_<double> Pi_single = Mat_<double>::zeros(S_r[i].size(), 1);

        //printf("(%d,%d)", S_r[i].size(), S_c[i].size());

        for (int j=0; j<Pij_single.rows; j++) {
            for (int k=0; k<Pij_single.cols; k++) {
                Pij_single(j,k) = 1.0/(double(Pij_single.cols)+1.0);
            }
            Pi_single(j,0) = 1.0/(double(Pij_single.cols)+1.0);
        }

        Pij.push_back(Pij_single.clone());
        Pij2.push_back(Pij_single.clone());
        Pi.push_back(Pi_single.clone());
        Pi2.push_back(Pi_single.clone());

    }

}

// New functions

int pTracking::find_matches_n(Mat Pij, Mat Pi, vector<Point2i> &matches) {

    qimshow(Pij);

    // for (int i=0; i<Pij.rows; i++) {

    //     LOG(INFO)<<sum(Pij.row(i));

    // }

    int count = 0;
    int count_single = 0;
    int count_mult = 0;
    int tiecount = 0;

    /*

    for (int i=0; i<Pij.size(); i++) {
        for (int j=0; j<Pij[i].size(); j++) {
            cout<<Pij[i][j]<<"\t";
        }
        cout<<endl;
    }



    vector< vector<int> > zematch;
    vector<int> container;

    for (int i=0; i<Pij.size(); i++)
        zematch.push_back(container);

    for (int i=0; i<Pij.size(); i++) {
        for (int j=0; j<Pij[i].size(); j++) {
            if (Pij[i][j]>0.99)
                zematch[S_r[i][j]].push_back(S_c[i][j]);
        }
    }

    for (int i=0; i<Pij.size(); i++) {

        vector<int> c;
        vector<int> cf;

        if (zematch[i].size()==0) {
            matches.push_back(Point2i(i,-1));
        } else {
            matches.push_back(Point2i(i,zematch[i][0]));
            count++;
        }

    }

    else {

            for (int j=0; j<zematch[i].size(); j++) {

                int counted=0;
                for (int k=0; k<c.size(); k++) {
                    if (zematch[i][j]==c[k]) {
                        counted=1;
                        cf[k]++;
                    }
                }
                if (counted==0) {
                    c.push_back(zematch[i][j]);
                    cf.push_back(1);
                }

            }

            // Finding maximum occurence frequency
            int modloc = 0;
            int modf = cf[0];
            for (int k=1; k<cf.size(); k++) {
                if (cf[k]>modf) {
                    modloc = k;
                    modf = cf[k];
                }
            }

            // Checking if multiple maxima exist
            int tie = 0;
            for (int k=0; k<cf.size(); k++) {

                if (k==modloc)
                    continue;

                if (cf[k]==cf[modloc]) {
                    tie=1;
                    tiecount++;
                    matches.push_back(Point2i(i,-2));
                    break;
                }

            }

            if (tie==0) {
                matches.push_back(Point2i(i,c[modloc]));
                count++;
                count_mult++;
            }

        }

        c.clear(); cf.clear();

    }

    */

    // LOG(INFO)<<"Ties: "<<tiecount<<", Matches: "<<count<<", Ratio: "<<double(count)/double(Pij.size())<<endl;

    LOG(INFO)<<"Matches: "<<count<<", Ratio: "<<double(count)/double(Pij.rows)<<endl;
    // LOG(INFO)<<"Singles: "<<count_single<<", Multiples: "<<count_mult;

    return(count);

}

double pTracking::update_probabilities_n(Mat &Pij, Mat &Pi, Mat &Pij2, Mat &Pi2) {

    Mat diffpij, diffpi;
    double difftotal = 0;
    Scalar diffsum;

    // for (int i=0; i<Pij.size(); i++) {

    //     for (int j=0; j<Pij[i].size(); j++) {
    //         difftotal += abs(Pij[i][j]-Pij2[i][j]);
    //         Pij[i][j] = Pij2[i][j];
    //     }

    //     difftotal += abs(Pi[i]-Pi2[i]);
    //     Pi[i] = Pi2[i];

    // }

    return(difftotal);

}

void pTracking::normalize_probabilites_n(Mat &Pij, Mat &Pi) {

    for (int i=0; i<Pij.rows; i++) {

        double s = 0;
        for (int j=0; j<Pij.cols; j++)
            s += Pij.at<double>(i,j);

        s += Pi.at<double>(i,0);

        for (int j=0; j<Pij.cols; j++)
            Pij.at<double>(i,j) /= s;

        Pi.at<double>(i,0) /= s;

        // for (int k=0; k<Pij[i].size(); k++) {
        //     Pij[i][k] /= sum;
        // }
        // Pi[i] /= sum;

    }

}

void pTracking::build_probability_sets_n(vector< vector<int> > S_r, vector< vector<int> > S_c, Mat &Pij, Mat &Pi, Mat &Pij2, Mat &Pi2) {

    for (int i=0; i<S_r.size(); i++) {

        for (int k=0; k<S_c[i].size(); k++) {
            Pij.at<double>(i,S_c[i][k]) = 1.0/(double(S_c[i].size())+1.0);
            Pij2.at<double>(i,S_c[i][k]) = 1.0/(double(S_c[i].size())+1.0);
        }
        Pi.at<double>(i,0) = 1.0/(double(S_c[i].size())+1.0);
        Pi2.at<double>(i,0) = 1.0/(double(S_c[i].size())+1.0);

    }

}

// ---

void pTracking::build_relaxation_sets(int frame1, int frame2, vector< vector<int> > S_r, vector< vector<int> > S_c, double C, double D, double E, double F, vector< vector< vector< vector<Point2i> > > > &theta) {

    vector<Point2i> theta_single;
    vector< vector<Point2i> > theta_j;
    vector< vector< vector<Point2i> > > theta_ij;
    Point3f dij, dkl;
    double dij_mag;

    for (int n=0; n<S_r.size(); n++) {
        for (int i=0; i<S_r[n].size(); i++) {
            for (int j=0; j<S_c[n].size(); j++) {
                for (int k=0; k<S_r[n].size(); k++) {
                    for (int l=0; l<S_c[n].size(); l++) {

                        dij = Point3f(all_points_[frame1][S_r[n][i]].x-all_points_[frame2][S_c[n][j]].x, all_points_[frame1][S_r[n][i]].y-all_points_[frame2][S_c[n][j]].y, all_points_[frame1][S_r[n][i]].z-all_points_[frame2][S_c[n][j]].z);

                        dkl = Point3f(all_points_[frame1][S_r[n][k]].x-all_points_[frame2][S_c[n][l]].x, all_points_[frame1][S_r[n][k]].y-all_points_[frame2][S_c[n][l]].y, all_points_[frame1][S_r[n][k]].z-all_points_[frame2][S_c[n][l]].z);

                        dij_mag = dist(all_points_[frame1][S_r[n][i]], all_points_[frame2][S_c[n][j]]);

                        /*
                        double thresh = E;
                        double ijmag = dist(dij, Point3f(0,0,0));
                        double klmag = dist(dkl, Point3f(0,0,0));
                        double ijkldot = dij.x*dkl.x + dij.y*dkl.y + dij.z*dkl.z;
                        if ((acos(ijkldot/(ijmag*klmag))*180.0/3.14159)<thresh) {
                            theta_single.push_back(Point2i(k,l));
                            //cout<<dist(dij, dkl)<<" ";
                            //cout<<dij_mag<<" ";
                            //cout<<acos(ijkldot/(ijmag*klmag))*180.0/3.14159<<" ";
                            //cout<<(dist(dij, dkl) < (E+(F*dij_mag)))<<endl;
                        }
                        */

                        double thresh;
                        if (method_==1)
                            thresh = E+(F*dij_mag);
                        else if (method_==2)
                            thresh = F*R_s;

                        if ( dist(dij, dkl) < thresh ) {
                            theta_single.push_back(Point2i(k,l));
                        }

                    }
                }
                theta_j.push_back(theta_single);
                theta_single.clear();
            }
            theta_ij.push_back(theta_j);
            theta_j.clear();
        }
        theta.push_back(theta_ij);
        theta_ij.clear();
    }

}

void pTracking::build_relaxation_sets_n(int frame1, int frame2, vector< vector<int> > S_r, vector< vector<int> > S_c, double C, double D, double E, double F, vector< vector< vector<Point2i> > > &theta) {

    vector<Point2i> theta_single;
    vector< vector<Point2i> > theta_j;
    //vector< vector< vector<Point2i> > > theta_ij;
    Point3f dij, dkl;
    double dij_mag;

    for (int i=0; i<S_r.size(); i++) {
        for (int j=0; j<S_c[i].size(); j++) {

            dij = Point3f(all_points_[frame1][i].x-all_points_[frame2][S_c[i][j]].x, all_points_[frame1][i].y-all_points_[frame2][S_c[i][j]].y, all_points_[frame1][i].z-all_points_[frame2][S_c[i][j]].z);

            for (int k=0; k<S_r[i].size(); k++) {
                for (int l=0; l<S_c[i].size(); l++) {

                    if (i==S_r[i][k] || S_c[i][j]==S_c[i][l])
                        continue;

                    //dij = Point3f(all_points_[frame1][i].x-all_points_[frame2][S_c[i][j]].x, all_points_[frame1][i].y-all_points_[frame2][S_c[i][j]].y, all_points_[frame1][i].z-all_points_[frame2][S_c[i][j]].z);

                    dkl = Point3f(all_points_[frame1][S_r[i][k]].x-all_points_[frame2][S_c[i][l]].x, all_points_[frame1][S_r[i][k]].y-all_points_[frame2][S_c[i][l]].y, all_points_[frame1][S_r[i][k]].z-all_points_[frame2][S_c[i][l]].z);

                    dij_mag = dist(all_points_[frame1][i], all_points_[frame2][S_c[i][j]]);

                    /*
                      double thresh = E;
                      double ijmag = dist(dij, Point3f(0,0,0));
                      double klmag = dist(dkl, Point3f(0,0,0));
                      double ijkldot = dij.x*dkl.x + dij.y*dkl.y + dij.z*dkl.z;
                      if ((acos(ijkldot/(ijmag*klmag))*180.0/3.14159)<thresh) {
                      theta_single.push_back(Point2i(k,l));
                      //cout<<dist(dij, dkl)<<" ";
                      //cout<<dij_mag<<" ";
                      //cout<<acos(ijkldot/(ijmag*klmag))*180.0/3.14159<<" ";
                      //cout<<(dist(dij, dkl) < (E+(F*dij_mag)))<<endl;
                      }
                    */

                    double thresh;
                    if (method_==1)
                        thresh = E+(F*dij_mag);
                    else if (method_==2)
                        thresh = F*R_s;

                    if ( dist(dij, dkl) < thresh ) {
                        theta_single.push_back(Point2i(S_r[i][k],S_c[i][l]));
                    }

                }
            }
            theta_j.push_back(theta_single);
            theta_single.clear();
        }
        theta.push_back(theta_j);
        theta_j.clear();
    }

}

vector< vector<int> > pTracking::neighbor_set(int frame, double r) {

    vector<int> indices;
    vector< vector<int> > indices_all;

    for (int i=0; i<all_points_[frame].size(); i++) {
        for (int j=0; j<all_points_[frame].size(); j++) {
            if (dist(all_points_[frame][i], all_points_[frame][j]) < r) {
                indices.push_back(j);
            }
        }
        indices_all.push_back(indices);
        indices.clear();
    }

    return(indices_all);

}

vector< vector<int> > pTracking::candidate_set(int frame1, int frame2, double r) {

    vector<int> indices;
    vector< vector<int> > indices_all;

    for (int i=0; i<all_points_[frame1].size(); i++) {
        for (int j=0; j<all_points_[frame2].size(); j++) {
            if (dist(all_points_[frame1][i], all_points_[frame2][j]) < r) {
                indices.push_back(j);
            }
        }
        indices_all.push_back(indices);
        indices.clear();
    }

    return(indices_all);

}

vector<int> pTracking::points_in_region(int frame, Point3f center, double r) {

    vector<int> indices;

    for (int i=0; i<all_points_[frame].size(); i++) {
        if (dist(all_points_[frame][i], center)<=r) indices.push_back(i);
    }

    return(indices);

}

void pTracking::find_long_paths(int l) {

    vector<int> path;
    for (int i=0; i<all_matches[0].size(); i++) {

        path.push_back(all_matches[0][i].x);

        int j=0;
        int p1=i;

        while(j < all_matches.size()) {
            if (all_matches[j][p1].y > -1) {
                path.push_back(all_matches[j][p1].y);
                p1 = all_matches[j][p1].y;
                j++;
            } else {
                break;
            }
        }

        //if (path.size()==all_points_.size()) {
        if (path.size()>l) {
            long_paths_.push_back(path);
            //cout<<path.size()<<endl;
        }

        path.clear();

    }
    cout<<"Paths longer than "<<l<<" frames: "<<long_paths_.size()<<endl;

}

void pTracking::find_sized_paths(int l) {

    vector< vector<int> > used;
    for (int i=0; i<all_points_.size(); i++) {
        vector<int> usub;
        used.push_back(usub);
    }

    particle_path pp;

    for (int k=0; k<all_matches.size()-l; k++) {
        for (int i=0; i<all_matches[k].size(); i++) {

            if (is_used(used, k, i)) {
                continue;
            } else {

                pp.path.push_back(all_matches[k][i].x);
                pp.start_frame = k;
                used[k].push_back(all_matches[k][i].x);

                int j=k;
                int p1=i;
                while(j < all_matches.size()) {
                    if (all_matches[j][p1].y > -1) {
                        pp.path.push_back(all_matches[j][p1].y);
                        used[j+1].push_back(all_matches[j][p1].y);
                            p1 = all_matches[j][p1].y;
                        j++;
                    } else {
                        break;
                    }
                }
                if (pp.path.size()>l) {
                    sized_paths_.push_back(pp);
                }
                pp.path.clear();
            }

        }
    }

    cout<<"Paths longer than "<<l<<" frames: "<<sized_paths_.size()<<endl;

}

void pTracking::write_quiver_data() {

    string qpath("");
    for (int i=0; i<path_.size()-4; i++) {
        qpath += path_[i];
    }
    qpath += "_quiver.txt";

    ofstream file;
    file.open(qpath.c_str());

    cout<<"Writing quiver data to: "<<qpath<<endl;

    file<<all_matches.size()<<endl;

    for (int frame=0; frame<all_matches.size(); frame++) {

        file<<match_counts[frame]<<endl;

        for (int i=0; i<all_matches[frame].size(); i++) {

            if (all_matches[frame][i].y > -1) {
                file<<all_points_[offset+frame][all_matches[frame][i].x].x<<"\t";
                file<<all_points_[offset+frame][all_matches[frame][i].x].y<<"\t";
                file<<all_points_[offset+frame][all_matches[frame][i].x].z<<"\t";

                double u = all_points_[offset+frame+1][all_matches[frame][i].y].x - all_points_[offset+frame][all_matches[frame][i].x].x;
                double v = all_points_[offset+frame+1][all_matches[frame][i].y].y - all_points_[offset+frame][all_matches[frame][i].x].y;
                double w = all_points_[offset+frame+1][all_matches[frame][i].y].z - all_points_[offset+frame][all_matches[frame][i].x].z;

                file<<u<<"\t"<<v<<"\t"<<w<<endl;
            }
        }

    }

    file.close();

}

void pTracking::write_tracking_result(string prefix) {

    string qpath("");
    for (int i=0; i<path_.size()-4; i++) {
        qpath += path_[i];
    }
    qpath += "_";
    qpath += prefix;
    qpath += "_result.txt";

    ofstream file;
    file.open(qpath.c_str());

    cout<<"Writing quiver data to: "<<qpath<<endl;

    file<<all_matches.size()<<endl;

    for (int frame=0; frame<all_matches.size(); frame++) {
        file<<all_matches[frame].size()<<endl;
        for (int i=0; i<all_matches[frame].size(); i++) {
            file<<all_matches[frame][i].x<<"\t"<<all_matches[frame][i].y<<endl;
        }
    }

    file.close();

}

void pTracking::write_all_paths(string path) {

    ofstream file;
    file.open(path.c_str());

    file<<all_matches.size()<<endl;

    for (int i=0; i<all_matches.size(); i++) {
        file<<match_counts[i]<<endl;
        for (int j=0; j<all_matches[i].size(); j++) {
            if (all_matches[i][j].y >= 0) {
                file<<all_points_[offset+i][all_matches[i][j].x].x<<"\t";
                file<<all_points_[offset+i][all_matches[i][j].x].y<<"\t";
                file<<all_points_[offset+i][all_matches[i][j].x].z<<"\t";
                file<<all_points_[offset+i+1][all_matches[i][j].y].x<<"\t";
                file<<all_points_[offset+i+1][all_matches[i][j].y].y<<"\t";
                file<<all_points_[offset+i+1][all_matches[i][j].y].z<<endl;
            }
        }
    }

    file.close();

}

void pTracking::write_long_quiver(string path, int l) {

    ofstream file;
    file.open(path.c_str());

    file<<l-1<<endl;

    for (int j=0; j<l-1; j++) {
        file<<long_paths_.size()<<endl;
        for (int i=0; i<long_paths_.size(); i++) {

            file<<all_points_[offset+j][long_paths_[i][j]].x<<"\t";
            file<<all_points_[offset+j][long_paths_[i][j]].y<<"\t";
            file<<all_points_[offset+j][long_paths_[i][j]].z<<"\t";

            double u = all_points_[offset+j+1][long_paths_[i][j+1]].x - all_points_[offset+j][long_paths_[i][j]].x;
            double v = all_points_[offset+j+1][long_paths_[i][j+1]].y - all_points_[offset+j][long_paths_[i][j]].y;
            double w = all_points_[offset+j+1][long_paths_[i][j+1]].z - all_points_[offset+j][long_paths_[i][j]].z;

            file<<u<<"\t"<<v<<"\t"<<w<<endl;
        }
    }
    file.close();

}

bool pTracking::is_used(vector< vector<int> > used, int k, int i) {

    for (int j=0; j<used[k].size(); j++) {
        if (used[k][j]==i)
            return 1;
    }

    return 0;

}

double pTracking::sim_performance() {

    double perf;

    for (int frame=0; frame<all_matches.size(); frame++) {
        int count = 0;
        int total = all_matches[frame].size();
        for (int i=0; i<all_matches[frame].size(); i++) {
            if(all_matches[frame][i].x==all_matches[frame][i].y)
                count++;
        }
        perf = double(count)/double(total);
    }

    //cout<<1-perf<<endl;

    return(1-perf);

}

vector<int> pTracking::get_match_counts() {

    return(match_counts);

}

Mat pTracking::getP() {

    return(P_.clone());

}

// Python wrapper
BOOST_PYTHON_MODULE(tracking) {

    using namespace boost::python;

    class_<pTracking>("pTracking", init<string, double, double>())
        .def("set_vars", &pTracking::set_vars)
        .def("track_frame", &pTracking::track_frame_n)
        .def("getP", &pTracking::getP)
        // .def("track_all", &pTracking::track_all)
        .def("get_match_counts", &pTracking::get_match_counts)
        .def("write_quiver_data", &pTracking::write_quiver_data)
    ;

}

/* LEGACY CODE (MOSTLY VISUALIZATION RELATED)
void pTracking::plot_long_paths() {

    PyVisualize vis;

    vis.figure3d();

    int count=0;
    int every = 1;
    for (int i=0; i<long_paths_.size(); i++) {
        if (i%every==0) {
        vector<Point3f> points;
        for (int j=0; j<long_paths_[i].size(); j++) {
            if (all_points_[j+offset][long_paths_[i][j]].x > -5 && all_points_[j+offset][long_paths_[i][j]].x < 45)
                points.push_back(all_points_[j+offset][long_paths_[i][j]]);
        }
        if (points.size()) {
            vis.plot3d(points, "k");
            count++;
        }
        points.clear();
        }
    }

    cout<<"Paths plotted: "<<count<<endl;

    vis.xlabel("x [mm]");
    vis.ylabel("y [mm]");
    vis.zlabel("z [mm]");
    vis.show();

}

void pTracking::plot_sized_paths() {

    PyVisualize vis;

    vis.figure3d();

    int count=0;
    int every = 1;
    for (int i=0; i<sized_paths_.size(); i++) {

        if (i%every==0) {
            vector<Point3f> points;
            for (int j=0; j<sized_paths_[i].path.size(); j++) {
                int sf = sized_paths_[i].start_frame;
                //if (all_points_[j+offset+sf][sized_paths_[i].path[j]].x > -5 && all_points_[j+offset+sf][sized_paths_[i].path[j]].x < 45)
                    points.push_back(all_points_[j+offset+sf][sized_paths_[i].path[j]]);
            }
            if (points.size()) {
                vis.plot3d(points, "k");
                count++;
            }
            points.clear();
        }
    }

    cout<<"Paths plotted: "<<count<<endl;

    vis.xlabel("x [mm]");
    vis.ylabel("y [mm]");
    vis.zlabel("z [mm]");
    vis.show();

}

void pTracking::plot_all_paths() {

    PyVisualize vis;

    vis.figure3d();

    for (int i=0; i<all_matches.size(); i++) {

        for (int j=0; j<all_matches[i].size(); j++) {

            if (all_matches[i][j].y >= 0) {
                vis.line3d(all_points_[offset+i][all_matches[i][j].x], all_points_[offset+i+1][all_matches[i][j].y]);
            }

        }

    }

    vis.xlabel("x [mm]");
    vis.ylabel("y [mm]");
    vis.zlabel("z [mm]");
    vis.show();

}
*/
