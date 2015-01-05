#ifndef TOOLS_LIBRARY
#define TOOLS_LIBRARY

#include "std_include.h"
#include "refocusing.h"
#include "rendering.h"
#include "typedefs.h"

using namespace std;
using namespace cv;

void init(int argc, char** argv);

void T_from_P(Mat P, Mat &H, double z, double scale, Size img_size);

bool dirExists(string dirPath);

int matrixMean(vector<Mat> mats_in, Mat &mat_out);

Mat P_from_KRT(Mat K, Mat rvec, Mat tvec, Mat rmean, Mat &P_u, Mat &P);

double dist(Point3f p1, Point3f p2);

void qimshow(Mat image);

void qimshow2(vector<Mat> imgs);

void pimshow(Mat image, double z, int n);

Mat getRotMat(double x, double y, double z);

void failureFunction();

void writeMat(Mat M, string path);

Mat getTransform(vector<Point2f> src, vector<Point2f> dst);

void listDir(string, vector<string> &);

void readImgStack(vector<string>, vector<Mat> &);

vector<double> linspace(double, double, int);

Mat cross(Mat_<double>, Mat_<double>);

Mat normalize(Mat_<double>);

saRefocus addCams(Scene, Camera, double, double, double);

// File IO class

class fileIO {

 public:
    ~fileIO() {
        LOG(INFO)<<"Closing file...";
        file.close(); 
    }

    fileIO(string filename);

    void operator<< (int);
    void operator<< (float);
    void operator<< (double);
    void operator<< (string);
    void operator<< (const char*);
    void operator<< (vector<int>);
    void operator<< (vector< vector<int> >);
    void operator<< (vector<float>);
    void operator<< (vector< vector<float> >);
    void operator<< (vector<double>);
    void operator<< (vector< vector<double> >);

    // TODO: add templated Mat data output to file
    // TODO: think of clean way to write image and vector of images to folders -> maybe write imageIO class

    /*
    void write(Mat);
    */

 protected:

 private:

    string getFilename(string filename);

    ofstream file;

};

class Movie {

 public:
    ~Movie() {}

    Movie(vector<Mat>);

 private:
    
    void play();
    void updateFrame();
    vector<Mat> frames_;
    int active_frame_;

};

#endif
