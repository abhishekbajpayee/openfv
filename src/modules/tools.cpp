#ifndef TOOLS_LIBRARY
#define TOOLS_LIBRARY

#include "std_include.h"
#include "typedefs.h"
#include "tools.h"

using namespace std;
using namespace cv;

void init(int argc, char** argv) {

    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    google::InstallFailureFunction(&failureFunction);
    FLAGS_logtostderr=1;

}

void T_from_P(Mat P, Mat &H, double z, double scale, Size img_size) {

    Mat_<double> A = Mat_<double>::zeros(3,3);

    for (int i=0; i<3; i++) {
        for (int j=0; j<2; j++) {
            A(i,j) = P.at<double>(i,j);
        }
    }

    for (int i=0; i<3; i++) {
        A(i,2) = P.at<double>(i,2)*z+P.at<double>(i,3);
    }

    Mat A_inv = A.inv();

    Mat_<double> D = Mat_<double>::zeros(3,3);
    D(0,0) = scale;
    D(1,1) = scale;
    D(2,2) = 1;
    D(0,2) = img_size.width*0.5;
    D(1,2) = img_size.height*0.5;
    
    Mat T = D*A_inv;
    
    H = T.clone();

}

bool dirExists(string dirPath) {

    if ( dirPath.c_str() == NULL) return false;

    DIR *pDir;
    bool bExists = false;

    pDir = opendir (dirPath.c_str());

    if (pDir != NULL)
    {
        bExists = true;    
        (void) closedir (pDir);
    }

    return bExists;

}

// Function to calculate mean of any matrix
// Returns 1 if success
int matrixMean(vector<Mat> mats_in, Mat &mat_out) {

    if (mats_in.size()==0) {
        cout<<"\nInput matrix vector empty!\n";
        return 0;
    }

    for (int i=0; i<mats_in.size(); i++) {
        for (int j=0; j<mats_in[0].rows; j++) {
            for (int k=0; k<mats_in[0].cols; k++) {
                mat_out.at<double>(j,k) += mats_in[i].at<double>(j,k);
            }
        }
    }

    mat_out = mat_out/double(mats_in.size());
    
    return 1;

}

// Construct aligned and unaligned P matrix from K, R and T matrices
Mat P_from_KRT(Mat K, Mat rvec, Mat tvec, Mat rmean, Mat &P_u, Mat &P) {

    Mat rmean_t;
    transpose(rmean, rmean_t);

    Mat R = rvec*rmean_t;
    
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            P_u.at<double>(i,j) = rvec.at<double>(i,j);
            P.at<double>(i,j) = R.at<double>(i,j);
        }
        P_u.at<double>(i,3) = tvec.at<double>(0,i);
        P.at<double>(i,3) = tvec.at<double>(0,i);
    }
    
    P_u = K*P_u;
    P = K*P;

}

double dist(Point3f p1, Point3f p2) {

    double distance = sqrt(pow(p2.x-p1.x,2) + pow(p2.y-p1.y,2) + pow(p2.z-p1.z,2));
    
    return(distance);

}

void qimshow(Mat image) {

    namedWindow("Image", CV_WINDOW_AUTOSIZE);
    imshow("Image", image);
    
    int key;
    while(1) {
        key = cvWaitKey(10);
        if ((key & 255) == 27)
            break;
    }
    destroyWindow("Image");

}

void pimshow(Mat image, double z, int n) {

    namedWindow("Image", CV_WINDOW_AUTOSIZE);
    
    char title[50];
    sprintf(title, "z = %f, n = %d", z, n);
    putText(image, title, Point(10,20), FONT_HERSHEY_PLAIN, 1.0, Scalar(255,0,0));
    imshow("Image", image);
    
    waitKey(0);
    destroyWindow("Image");

}

// Yaw Pitch Roll Rotation Matrix
Mat getRotMat(double x, double y, double z) {

    x = x*pi/180.0;
    y = y*pi/180.0;
    z = z*pi/180.0;

    Mat_<double> Rx = Mat_<double>::zeros(3,3);
    Mat_<double> Ry = Mat_<double>::zeros(3,3);
    Mat_<double> Rz = Mat_<double>::zeros(3,3);

    Rx(0,0) = 1;
    Rx(1,1) = cos(x);
    Rx(1,2) = -sin(x);
    Rx(2,1) = sin(x);
    Rx(2,2) = cos(x);

    Ry(0,0) = cos(y);
    Ry(1,1) = 1;
    Ry(2,0) = -sin(y);
    Ry(2,2) = cos(y);
    Ry(0,2) = sin(y);

    Rz(0,0) = cos(z);
    Rz(0,1) = -sin(z);
    Rz(1,0) = sin(z);
    Rz(1,1) = cos(z);
    Rz(2,2) = 1;

    Mat R = Rz*Ry*Rx;

    return(R);

}

void failureFunction() {

    LOG(INFO)<<"Good luck debugging that X-|";
    exit(1);

}

void writeMat(Mat M, string path) {

    ofstream file;
    file.open(path.c_str());

    Mat_<double> A = M;

    for (int i=0; i<M.rows; i++) {
        for (int j=0; j<M.cols; j++) {
            file<<A.at<double>(i,j)<<"\t";
        }
        file<<"\n";
    }

    VLOG(3)<<"Written matrix to file "<<path<<endl;

    file.close();

}

Mat getTransform(vector<Point2f> src, vector<Point2f> dst) {

    Mat_<double> A1 = Mat_<double>::zeros(8,8);
    Mat_<double> B1 = Mat_<double>::zeros(8,1);
    for (int i=0; i<4; i++) {
        A1(i*2,0) = src[i].x; A1(i*2,1) = src[i].y; A1(i*2,2) = 1;
        A1(i*2+1,3) = src[i].x; A1(i*2+1,4) = src[i].y; A1(i*2+1,5) = 1;
        A1(i*2,6) = -dst[i].x*src[i].x; A1(i*2,7) = -dst[i].x*src[i].y;
        A1(i*2+1,6) = -dst[i].y*src[i].x; A1(i*2+1,7) = -dst[i].y*src[i].y;
        B1(i*2,0) = dst[i].x;
        B1(i*2+1,0) = dst[i].y;
    }

    Mat A1t;
    transpose(A1, A1t);

    Mat C1;
    invert(A1t*A1, C1, DECOMP_SVD);
    Mat C2 = A1t*B1;

    Mat_<double> R = C1*C2;

    Mat_<double> H = Mat_<double>::zeros(3,3);
    H(0,0) = R(0,0); H(0,1) = R(1,0); H(0,2) = R(2,0);
    H(1,0) = R(3,0); H(1,1) = R(4,0); H(1,2) = R(5,0);
    H(2,0) = R(6,0); H(2,1) = R(7,0); H(2,2) = 1.0;


    // old


    Mat M(3, 3, CV_64F), X(8, 1, CV_64F, M.data);
    double a[8][8], b[8];
    Mat A(8, 8, CV_64F, a), B(8, 1, CV_64F, b);

    for( int i = 0; i < 4; ++i )
    {
        a[i][0] = a[i+4][3] = src[i].x;
        a[i][1] = a[i+4][4] = src[i].y;
        a[i][2] = a[i+4][5] = 1;
        a[i][3] = a[i][4] = a[i][5] =
        a[i+4][0] = a[i+4][1] = a[i+4][2] = 0;
        a[i][6] = -src[i].x*dst[i].x;
        a[i][7] = -src[i].y*dst[i].x;
        a[i+4][6] = -src[i].x*dst[i].y;
        a[i+4][7] = -src[i].y*dst[i].y;
        b[i] = dst[i].x;
        b[i+4] = dst[i].y;
    }

    solve( A, B, X, DECOMP_SVD );
    ((double*)M.data)[8] = 1.;

    return M;

}

void listDir(string path, vector<string> &files) {
    
    DIR *dir;
    struct dirent *ent;

    string temp_name;
    
    dir = opendir(path.c_str());
    while(ent = readdir(dir)) {
        temp_name = ent->d_name;
        if (temp_name.compare(".")) {
            if (temp_name.compare("..")) {
                string path_file = path+temp_name;
                files.push_back(path_file);
            }
        }
    }

}

void readImgStack(vector<string> img_names, vector<Mat> &imgs) {

    for (int i=0; i<img_names.size(); i++) {
        Mat img = imread(img_names[i], 0);
        imgs.push_back(img);
    }

}

vector<double> linspace(double a, double b, int n) {
    
    vector<double> array;
    double step = (b-a) / (n-1);

    while(a <= b) {
        array.push_back(a);
        a += step;           // could recode to better handle rounding errors
    }
    return array;

}

// returns normalized(A x B) where A and B are 2 vectors
Mat cross(Mat_<double> A, Mat_<double> B) {

    Mat_<double> result = Mat_<double>::zeros(3,1);

    result(0,0) = A(1,0)*B(2,0) - A(2,0)*B(1,0);
    result(1,0) = A(2,0)*B(0,0) - A(0,0)*B(2,0);
    result(2,0) = A(0,0)*B(1,0) - A(1,0)*B(0,0);

    return(normalize(result));

}

// Normalizes a vector and then returns it
Mat normalize(Mat_<double> A) {

    double d = sqrt(pow(A(0,0), 2) + pow(A(1,0), 2) + pow(A(2,0), 2));
    A(0,0) /= d; A(1,0) /= d; A(2,0) /= d;
    return(A);

}

// BOOST_PYTHON_MODULE(libsaTools) {

//     using namespace boost::python;
//     def("dirExists", dirExists);

// }

#endif
