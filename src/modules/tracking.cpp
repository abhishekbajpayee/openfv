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

void pTracking::initialize() {

    N = 500;
    tol = 0.1;

    V_n = (4/3)*pi*pow(R_n,3);
    V_s = (4/3)*pi*pow(R_s,3);

    //cout<<"R_n: "<<R_n<<", R_s: "<<R_s<<endl;

    A = 0.3;
    B = 3.0;
    C = 0.1;
    D = 5.0;
    E = 1.0;
    F = 0.05;

}

void pTracking::set_vars(double rn, double rs, double a, double b, double c, double d, double e, double f) {

    R_n = rn;
    R_s = rs;

    A = a;
    B = b;
    C = c;
    D = d;
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

    //cout<<"Reading points to track...";
    
    int num_frames;
    file>>num_frames;

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

        vol.v = (vol.x2-vol.x1)*(vol.y2-vol.y1)*(vol.z2-vol.z1);
        double density = points_.size()/vol.v;
        double complexity = 300*points_.size()*pow(density, 4)*pow(R_s*R_n,6);
        //cout<<"complexity: "<<complexity<<endl;
        all_points_.push_back(points_);
        vols_.push_back(vol);
        points_.clear();

    }

    //cout<<"done!"<<endl;

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

    //cout<<"Matching frames "<<f1<<" and "<<f2<<" | ";

    int n1, n2;
    n1 = all_points_[f1].size();
    n2 = all_points_[f2].size();

    //cout<<"Neighbor Sets...";
    vector< vector<int> > S_r = neighbor_set(f1, f1, R_n);
    vector< vector<int> > S_c = neighbor_set(f1, f2, R_s);

    vector<Mat> Pij, Pi, Pij2, Pi2;
    //cout<<"Probability Sets...";
    build_probability_sets(S_r, S_c, Pij, Pi, Pij2, Pi2);

    //cout<<"Relaxation Sets...\n";
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

    //cout<<"Final residual change: "<<diff<<" in "<<n+1<<" iterations. "<<endl;

    vector<Point2i> matches;
    count = find_matches(Pij, Pi, S_r, S_c, matches);

    return(matches);

}

int pTracking::find_matches(vector<Mat> Pij, vector<Mat> Pi, vector< vector<int> > S_r, vector< vector<int> > S_c, vector<Point2i> &matches) {

    vector< vector<int> > zematch;
    vector<int> container;
    for (int i=0; i<Pij.size(); i++)
        zematch.push_back(container);

    for (int i=0; i<Pij.size(); i++) {

        for (int j=0; j<Pij[i].rows; j++) {
            for (int k=0; k<Pij[i].cols; k++) {

                if (Pij[i].at<double>(j,k)>0.99) {
                    zematch[S_r[i][j]].push_back(S_c[i][k]);
                }

            }
        }

    }

    int count = 0;
    int tiecount = 0;
    for (int i=0; i<Pij.size(); i++) {
        
        vector<int> c;
        vector<int> cf;

        if (zematch[i].size()==0) {
            matches.push_back(Point2i(i,-1));
        } else if (zematch[i].size()==1) {
            matches.push_back(Point2i(i,zematch[i][0]));
            count++;
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
            }

        }

        c.clear(); cf.clear();

    }

    //cout<<"Ties: "<<tiecount<<", Matches: "<<count<<", Ratio: "<<double(count)/double(Pij.size())<<endl;
    
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
                        double thresh = 20;
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

                        double thresh = E+(F*dij_mag);
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

vector< vector<int> > pTracking::neighbor_set(int frame1, int frame2, double r) {

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

void pTracking::plot_complete_paths() {

    PyVisualize vis;

    vis.figure3d();

    vector<int> path;
    vector< vector<int> > all_paths;
    for (int i=0; i<all_points_[0].size(); i++) {
        
        path.push_back(i);
        
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
        if (path.size()>20) {
            all_paths.push_back(path);
            //cout<<path.size()<<endl;
        }

        path.clear();

    }
    cout<<"full: "<<all_paths.size()<<endl;
    
    int every = 3;
    for (int i=0; i<all_paths.size(); i++) {
        if (i%every==1) {
        vector<Point3f> points;
        for (int j=0; j<all_paths[i].size(); j++) {
            points.push_back(all_points_[j+offset][all_paths[i][j]]);
        }
        vis.plot3d(points, "k");
        points.clear();
        }
    }
    
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

    return(perf);

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
