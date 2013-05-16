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

#include "std_include.h"
#include "refocusing.h"
#include "tracking.h"
#include "typedefs.h"
#include "tools.h"

using namespace std;
using namespace cv;

void pTracking::read_points(string path) {
    
    Point3f point;
    volume vol;

    ifstream file;
    file.open(path.c_str());

    cout<<"\nReading points to track...\n";
    
    int num_frames = 30;

    for (int i=0; i<num_frames; i++) {

        int num_points;
        file>>num_points;

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

        all_points_.push_back(points_);
        vols_.push_back(vol);
        points_.clear();

    }

    cout<<"done!\n";

}

void pTracking::track() {

    int f1, f2;
    f1 = 0;
    f2 = 1;

    double A, B, C, D, E, F;
    A = 0.3;
    B = 3.0;
    C = 0.1;
    D = 5.0;
    E = 1.0;
    F = 0.05;

    double R_n, R_s;
    R_n = 20.0;
    R_s = 20.0;

    int n1, n2;
    n1 = all_points_[f1].size();
    n2 = all_points_[f2].size();

    cout<<"Neighbor sets...\n";
    vector< vector<int> > S_r = neighbor_set(f1, f1, R_n);
    vector< vector<int> > S_c = neighbor_set(f1, f2, R_s);

    /*
    ofstream file;
    file.open("../temp/window1.txt");
    for (int i=0; i<x.size(); i++) {
        file<<all_points_[0][x[i]].x<<"\t";
        file<<all_points_[0][x[i]].y<<"\t";
        file<<all_points_[0][x[i]].z<<endl;
    }
    file.close();
    file.open("../temp/window2.txt");
    for (int i=0; i<y.size(); i++) {
        file<<all_points_[1][y[i]].x<<"\t";
        file<<all_points_[1][y[i]].y<<"\t";
        file<<all_points_[1][y[i]].z<<endl;
    }
    file.close();
    */

    vector<Mat> Pij, Pi, Pij2, Pi2;
    cout<<"Probability sets...\n";
    build_probability_sets(S_r, S_c, Pij, Pi, Pij2, Pi2);

    cout<<"Relaxation sets...\n";
    vector< vector<Point2i> > theta;
    build_relaxation_sets(f1, f2, S_r, S_c, C, D, E, F, theta);
    
    cout<<"solving...";

    int N = 5;

    for (int n=0; n<N; n++) {
        
        double diff=0;
        //cout<<Pij[0]<<endl;

        for (int k=0; k<1; k++) {

            for (int i=0; i<Pij[k].rows; i++) {
                for (int j=0; j<Pij[k].cols; j++) {

                    double sum = 0;
                    for (int l=0; l<theta[k].size(); l++) {
                        sum += Pij[k].at<double>(theta[k][l].x, theta[k][l].y);
                    }
                    cout<<"sum: "<<sum<<endl;

                    //cout<<Pij[k].at<double>(i,j)<<" ";
                    Pij2[k].at<double>(i,j) = Pij[k].at<double>(i,j)*(A + (B*sum));
                    diff += abs(Pij2[k].at<double>(i,j)-Pij[k].at<double>(i,j));
                    cout<<Pij[k].at<double>(i,j)<<" "<<Pij2[k].at<double>(i,j)<<endl;

                }
            }

        }

        //cout<<n<<": "<<diff<<endl;

        normalize_probabilites(Pij2,Pi2);

        Pij = Pij2;
        Pi = Pi2;

    }

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

void pTracking::build_relaxation_sets(int frame1, int frame2, vector< vector<int> > S_r, vector< vector<int> > S_c, double C, double D, double E, double F, vector< vector<Point2i> > &theta) {

    vector<Point2i> theta_single;
    Point3f dij, dkl;
    double dij_mag;

    for (int n=0; n<S_r.size(); n++) {
        for (int i=0; i<S_r[n].size(); i++) {
            for (int j=0; j<S_c[n].size(); j++) {
                for (int k=0; k<S_r[n].size(); k++) {
                    for (int l=0; l<S_c[n].size(); l++) {
                        dij = Point3f(all_points_[frame1][i].x-all_points_[frame2][k].x, all_points_[frame1][i].y-all_points_[frame2][k].y, all_points_[frame1][i].z-all_points_[frame2][k].z);
                        dkl = Point3f(all_points_[frame1][k].x-all_points_[frame2][l].x, all_points_[frame1][k].y-all_points_[frame2][l].y, all_points_[frame1][k].z-all_points_[frame2][l].z);
                        dij_mag = dist(all_points_[frame1][i], all_points_[frame2][j]);
                        if ( dist(dij, dkl) < (E+(F*dij_mag)) ) theta_single.push_back(Point2i(k,l));
                    }
                }
            }
        }
        theta.push_back(theta_single);
        theta_single.clear();
    }

}

vector< vector<int> > pTracking::neighbor_set(int frame1, int frame2, double r) {

    vector<int> indices;
    vector< vector<int> > indices_all;

    for (int i=0; i<all_points_[frame1].size(); i++) {
        for (int j=0; j<all_points_[frame2].size(); j++) {
            if (dist(all_points_[frame1][i], all_points_[frame2][j]) < r) indices.push_back(j);
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
