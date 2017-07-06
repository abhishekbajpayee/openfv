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

    void plot(vector<double> x, vector<double> y, string args);
    void plot3d(vector<Point3f> points, string args);
    void line3d(Point3f p1, Point3f p2);

    void scatter3d(vector<Point3f> points, vector<int> indices, string size, string color);
    void scatter3d(vector<Point3f> points, string size, string color);

    void string_from_vdouble(vector<double> p, string &str);
    void string_from_vPoint3f(vector<Point3f> points, vector<int> indices, vector<string> &strs);
    void string_from_vPoint3f(vector<Point3f> points, vector<string> &strs);

    void xlabel(string label);
    void ylabel(string label);
    void zlabel(string label);
    void title(string label);
    void figure3d();
    void hold(int value);
    void clear();
    void show();

 private:

    int hold_;
    int figure_;

};

#endif
