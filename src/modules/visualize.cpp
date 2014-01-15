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

// ACTUAL PLOTTING FUNCTIONS

void PyVisualize::plot(vector<double> x, vector<double> y, string args) {
    
    string sx, sy;
    string_from_vdouble(x, sx);
    string_from_vdouble(y, sy);

    string call("pl.plot(");
    call += sx + "," + sy + ",'" + args + "')";

    PyRun_SimpleString(call.c_str());

}

void PyVisualize::plot3d(vector<Point3f> points, string args) {
    
    vector<string> strs;
    string_from_vPoint3f(points, strs);

    string call("pl.plot(");
    call += strs[0] + "," + strs[1] + "," + strs[2] + ",'" + args + "',linewidth=0.5)";

    PyRun_SimpleString(call.c_str());
    //PyRun_SimpleString("axis('equal')");

}

void PyVisualize::line3d(Point3f p1, Point3f p2) {
    
    vector<Point3f> points;
    points.push_back(p1);
    points.push_back(p2);
    vector<string> strs;
    
    string_from_vPoint3f(points, strs);

    string call("ax.plot(");
    call += strs[0] + "," + strs[1] + "," + strs[2] + ",'k',linewidth=0.5)";

    PyRun_SimpleString(call.c_str());

}

void PyVisualize::scatter3d(vector<Point3f> points, vector<int> indices, string size, string color) {

    vector<string> strs;
    string_from_vPoint3f(points, indices, strs);

    string call("ax.scatter(");
    call += strs[0] + "," + strs[1] + "," + strs[2];
    call += ",s=" + size;
    call += ",c='" + color + "')";

    PyRun_SimpleString(call.c_str());

}

void PyVisualize::scatter3d(vector<Point3f> points, string size, string color) {

    vector<string> strs;
    string_from_vPoint3f(points, strs);

    string call("ax.scatter(");
    call += strs[0] + "," + strs[1] + "," + strs[2];
    call += ",s=" + size;
    call += ",c='" + color + "')";

    PyRun_SimpleString(call.c_str());

}

// STRING CONSTRUCTION FUNCTIONS

void PyVisualize::string_from_vdouble(vector<double> p, string &str) {

    stringstream sp;

    sp<<"["<<p[0];

    for (int i=1; i<p.size(); i++) {
        sp<<","<<p[i];
    }

    sp<<"]";

    str = sp.str();

}
    

void PyVisualize::string_from_vPoint3f(vector<Point3f> points, vector<int> indices, vector<string> &strs) {

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

    strs.push_back(sx.str());
    strs.push_back(sy.str());
    strs.push_back(sz.str());

}

void PyVisualize::string_from_vPoint3f(vector<Point3f> points, vector<string> &strs) {

    stringstream sx,sy,sz;

    sx<<"["<<points[0].x;
    sy<<"["<<points[0].y;
    sz<<"["<<points[0].z;

    for (int i=1; i<points.size(); i++) {
        sx<<","<<points[i].x;
        sy<<","<<points[i].y;
        sz<<","<<points[i].z;
    }

    sx<<"]";
    sy<<"]";
    sz<<"]";

    strs.push_back(sx.str());
    strs.push_back(sy.str());
    strs.push_back(sz.str());

}

// PLOTTING TOOLS

void PyVisualize::xlabel(string label) {

    string call("pl.xlabel('");
    call += label + "')";
    PyRun_SimpleString(call.c_str());

}

void PyVisualize::ylabel(string label) {

    string call("pl.ylabel('");
    call += label + "')";
    PyRun_SimpleString(call.c_str());

}

void PyVisualize::zlabel(string label) {

    string call("pl.zlabel('");
    call += label + "')";
    PyRun_SimpleString(call.c_str());

}

void PyVisualize::title(string label) {

    string call("pl.title('");
    call += label + "')";
    PyRun_SimpleString(call.c_str());

}

void PyVisualize::figure3d() {

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
