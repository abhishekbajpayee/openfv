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

void addCams(Scene, Camera, double, double, double, saRefocus&);

void addCams4(Scene, Camera, double, double, double, saRefocus&);

void saveScene(string filename, Scene scn);

void loadScene(string filename, Scene &scn);

Scene loadScene(string filename);

vector<double> vortex(double, double, double, double);

// Movie class

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

// File IO class

class fileIO {

 public:
    ~fileIO() {
        LOG(INFO)<<"Closing file...";
        file.close(); 
    }

    fileIO(string filename);

    fileIO& operator<< (int);
    fileIO& operator<< (float);
    fileIO& operator<< (double);
    fileIO& operator<< (string);
    fileIO& operator<< (const char*);
    fileIO& operator<< (vector<int>);
    fileIO& operator<< (vector< vector<int> >);
    fileIO& operator<< (vector<float>);
    fileIO& operator<< (vector< vector<float> >);
    fileIO& operator<< (vector<double>);
    fileIO& operator<< (vector< vector<double> >);
    fileIO& operator<< (Mat);

    // TODO: add templated Mat data output to file

    /*
    void write(Mat);
    */

 protected:

 private:

    string getFilename(string filename);

    ofstream file;

};

class imageIO {

 public:
    ~imageIO() {

    }

    imageIO(string path);

    void setPrefix(string prefix);

    void operator<< (Mat);
    void operator<< (vector<Mat>);

 protected:

 private:
    
    string dir_path_;
    string prefix_;
    string ext_;

    int counter_;

    int DIR_CREATED;

};

#endif
