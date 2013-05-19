// -------------------------------------------------------
// -------------------------------------------------------
// Synthetic Aperture - Particle Tracking Velocimetry Code
// --- Visualization Library Header ---
// -------------------------------------------------------
// Author: Abhishek Bajpayee
//         Dept. of Mechanical Engineering
//         Massachusetts Institute of Technology
// -------------------------------------------------------
// -------------------------------------------------------

#ifndef VISUALIZATION_LIBRARY
#define VISUALIZATION_LIBRARY

#include "std_include.h"
#include "tools.h"

using namespace cv;
using namespace std;

class PyVisualize {

 public:
    ~PyVisualize() {
        Py_Exit(0);
    }

    PyVisualize() {
        Py_Initialize();
        PyRun_SimpleString("import pylab as pl");
        PyRun_SimpleString("from mpl_toolkits.mplot3d import Axes3D");
        hold_ = 0;
        figure_ = 0;
    }

    void line3d(Point3f p1, Point3f p2);
    void scatter3d(vector<Point3f> points, vector<int> indices, string size, string color);
    void figure();
    void hold(int value);
    void clear();
    void show();

 private:

    int hold_;
    int figure_;

};

#endif
