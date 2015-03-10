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
//#include "refocusing.h"
#include "tracking.h"
#include "typedefs.h"
#include "visualize.h"
#include "tools.h"

using namespace std;
using namespace cv;

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

void pTracking::write_tracking_result() {

    string qpath("");
    for (int i=0; i<path_.size()-4; i++) {
        qpath += path_[i];
    }
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
