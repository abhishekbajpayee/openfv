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
#include "pLoc.h"

using namespace std;
using namespace cv;

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
    
    /*
    for (int i=0; i<points_out.size(); i++) {
        points_out[i].x++;
        points_out[i].y++;
    }
    */

}

void pLocalize::refine_subpixel(Mat image, vector<Point2f> points_in, vector<Point2f> &points_out) {

    int h = image.rows;
    int w = image.cols;

    double x_num, x_den, y_num, y_den, x_spix, y_spix;

    for (int i=0; i<points_in.size(); i++) {
        
        x_num=0;
        y_num=0;
        x_den=0;
        y_den=0;

        for (int x=points_in[i].x-window_; x<=points_in[i].x+window_; x++) {
            for (int y=points_in[i].y-window_; y<=points_in[i].y+window_; y++) {

                if (x<0 || x>=w || y<0 || y>=h) continue;
                
                Scalar intensity = image.at<uchar>(y,x);
                int i_xy = intensity.val[0];
                x_num = x_num + double(i_xy)*(x+0.5);
                y_num = y_num + double(i_xy)*(y+0.5);
                x_den = x_den + double(i_xy);
                y_den = y_den + double(i_xy);

            }
        }

        Point2f point;
        point.x = x_num/x_den;
        point.y = y_num/y_den;
        points_out.push_back(point);

    }

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
