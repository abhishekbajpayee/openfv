// -------------------------------------------------------
// -------------------------------------------------------
// Synthetic Aperture - Particle Tracking Velocimetry Code
// --- Visualization Library ---
// -------------------------------------------------------
// Author: Abhishek Bajpayee
//         Dept. of Mechanical Engineering
//         Massachusetts Institute of Technology
// -------------------------------------------------------
// -------------------------------------------------------

#include "std_include.h"
#include "visualize.h"
#include "tools.h"

using namespace cv;
using namespace std;

void PyVisualize::line3d(Point3f p1, Point3f p2) {

    if (hold_==0) {
        //PyRun_SimpleString("ax = fig.add_subplot(111, projection='3d')");
    }
    
    string call("ax.plot(");

    stringstream sx,sy,sz;

    sx<<"["<<p1.x<<","<<p2.x<<"]";
    sy<<"["<<p1.y<<","<<p2.y<<"]";
    sz<<"["<<p1.z<<","<<p2.z<<"]";

    call += sx.str() + "," + sy.str() + "," + sz.str();
    //call += ",s=" + size;
    //call += ",c='" + color + "')";
    call += ")";

    PyRun_SimpleString(call.c_str());

}

void PyVisualize::scatter3d(vector<Point3f> points, vector<int> indices, string size, string color) {

    if (hold_==0) {
        //PyRun_SimpleString("ax = fig.add_subplot(111, projection='3d')");
    }

    string call("ax.scatter(");

    stringstream sx,sy,sz;

    sx<<"["<<points[indices[0]].x;
    sy<<"["<<points[indices[0]].y;
    sz<<"["<<points[indices[0]].z;

    for (int i=1; i<indices.size(); i++) {
        sx<<","<<points[indices[i]].x;
        sy<<","<<points[indices[i]].y;
        sz<<","<<points[indices[i]].z;
    }

    sx<<"]";
    sy<<"]";
    sz<<"]";

    call += sx.str() + "," + sy.str() + "," + sz.str();
    call += ",s=" + size;
    call += ",c='" + color + "')";

    PyRun_SimpleString(call.c_str());

}

void PyVisualize::figure() {

    PyRun_SimpleString("fig = pl.figure()");
    PyRun_SimpleString("ax = fig.add_subplot(111, projection='3d')");
    figure_ = 1;

}

void PyVisualize::hold(int value) {

    hold_ = value;

}

void PyVisualize::clear() {

    PyRun_SimpleString("pl.close()");
    hold_ = 0;

}

void PyVisualize::show() {

    PyRun_SimpleString("pl.show()");

}
