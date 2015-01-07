// OpenCV headers
#include <cv.h>
#include <highgui.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

// Standard headers
#include <iostream>
#include <string>
#include <stdio.h>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <math.h>
#include <algorithm>
#include <ctime>
#include <sys/stat.h>
#include <omp.h>

// Tiff library
#include <tiffio.h>

// Ceres Solver headers
#include <ceres/ceres.h>
#include <ceres/rotation.h>

// Profiler header
#include <gperftools/profiler.h>

// Boost libraries
#include <boost/program_options.hpp>
#include <boost/python.hpp>
#include <boost/filesystem.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

// Python library
#include <Python.h>

