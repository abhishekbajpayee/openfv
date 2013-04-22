#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include "std_include.h"

using namespace cv;
using namespace std;

// CLASS AND STRUCT DEFINITIONS

// Read a Bundle Adjustment dataset
class baProblem {
 public:
    ~baProblem() {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] plane_index_;
        delete[] observations_;
        delete[] parameters_;
    }

    int num_observations()       const { return num_observations_;               }
    const double* observations() const { return observations_;                   }
    double* mutable_cameras()          { return parameters_;                     }
    double* mutable_points()           { return parameters_ + 9 * num_cameras_; }
    double* mutable_planes()           { return parameters_ + 9 * num_cameras_ + 3 * num_points_; }
    
    int num_cameras()                  { return num_cameras_;                    }
    int num_points()                   { return num_points_;                     }
    int num_planes()                   { return num_planes_;                     }
    int* camera_index()                { return camera_index_;                   }
    int* point_index()                 { return point_index_;                    }
    
    double* mutable_camera_for_observation(int i) {
        return mutable_cameras() + camera_index_[i] * 9;
    }
    double* mutable_point_for_observation(int i) {
        return mutable_points() + point_index_[i] * 3;
    }
    double* mutable_plane_for_observation(int i) {
        return mutable_planes() + plane_index_[i] * 4;
    }
    
    bool LoadFile(const char* filename) {
        FILE* fptr = fopen(filename, "r");
        if (fptr == NULL) {
            return false;
        };
        
        FscanfOrDie(fptr, "%d", &num_cameras_);
        FscanfOrDie(fptr, "%d", &num_planes_);
        FscanfOrDie(fptr, "%d", &num_points_);
        FscanfOrDie(fptr, "%d", &num_observations_);
        
        point_index_ = new int[num_observations_];
        camera_index_ = new int[num_observations_];
        plane_index_ = new int[num_observations_];
        observations_ = new double[2 * num_observations_];
        
        num_parameters_ = (9 * num_cameras_) + (3 * num_points_) + (4 * num_planes_);
        parameters_ = new double[num_parameters_];
        
        for (int i = 0; i < num_observations_; ++i) {
            FscanfOrDie(fptr, "%d", camera_index_ + i);
            FscanfOrDie(fptr, "%d", plane_index_ + i);
            FscanfOrDie(fptr, "%d", point_index_ + i);
            for (int j = 0; j < 2; ++j) {
                FscanfOrDie(fptr, "%lf", observations_ + 2*i + j);
            }
        }
        
        for (int i = 0; i < num_parameters_; ++i) {
            FscanfOrDie(fptr, "%lf", parameters_ + i);
            
        }
        return true;
    }

    double cx;
    double cy;
    double scale;
    
 private:
    template<typename T>
        void FscanfOrDie(FILE *fptr, const char *format, T *value) {
        int num_scanned = fscanf(fptr, format, value);
        if (num_scanned != 1) {
            LOG(FATAL) << "Invalid UW data file.";
        }
    }
    
    int num_cameras_;
    int num_planes_;
    int num_points_;
    int* point_index_;
    int* camera_index_;
    int* plane_index_;
    int num_observations_;
    int num_parameters_;
    
    double* observations_;
    double* parameters_;
};

// Read a Bundle Adjustment alignment dataset
class alignProblem {
 public:
    ~alignProblem() {
        delete[] parameters_;
        delete[] kr;
        delete[] r_c_;
        //delete[] t_c_;
    }

    double* mutable_params()           { return parameters_;    }
    double* constants_r()              { return kr;             }
    double* r_c()                      { return r_c_;           }
    //double* t_c()                      { return t_c_;           }
    
    int num_cameras()                  { return num_cameras_;   }
    /*
    double* mutable_params_for_camera(int i) {
        return mutable_params() + i*6;
    }
    double* constants_for_camera(int i) {
        return constants() + (i*6);
    }
    */
    bool LoadFile(const char* filename) {
        FILE* fptr = fopen(filename, "r");
        if (fptr == NULL) {
            return false;
        };
        
        FscanfOrDie(fptr, "%d", &num_cameras_);
        
        num_parameters_ = (6 * num_cameras_);
        parameters_ = new double[num_parameters_];

        kr = new double[3 * (num_cameras_-1)];
        
        for (int i = 0; i < num_parameters_; ++i) {
            FscanfOrDie(fptr, "%lf", parameters_ + i);
        }
        return true;
    }

    bool initialize() {
        
        r_c_ = new double[3];

        for (int i=0; i<3; i++) {
            r_c_[i] = 0;
        }

        // Calculate original aperture center parameters
        for (int i=0; i<num_cameras_; i++) {
            for (int j=0; j<3; j++) {
                r_c_[j] += (parameters_[(i*6)+j])/double(num_cameras_);
            }
        }
        cout<<"mean r: "<<r_c_[0]<<" "<<r_c_[1]<<" "<<r_c_[2]<<endl;

        // Initial difference between Ri and Rm for all cameras
        for (int i=1; i<num_cameras_; i++) {
            for (int j=0; j<3; j++) {
                kr[((i-1)*3)+j] = parameters_[j] - parameters_[(i*6)+j];
            }
        }

        return true;
    }
    
 private:
    template<typename T>
        void FscanfOrDie(FILE *fptr, const char *format, T *value) {
        int num_scanned = fscanf(fptr, format, value);
        if (num_scanned != 1) {
            LOG(FATAL) << "Invalid UW data file.";
        }
    }
    
    int num_cameras_;
    int num_parameters_;
    
    double* parameters_;
    double* kr; 
    double* r_c_;
    //double* t_c_;

};

// Pinhole Reprojection Error function
struct pinholeReprojectionError {
pinholeReprojectionError(double observed_x, double observed_y, double cx, double cy, int num_cams)
: observed_x(observed_x), observed_y(observed_y), cx(cx), cy(cy), num_cams(num_cams) {}
    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    const T* const plane,
                    T* residuals) const {
        
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        
        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        
        // Compute the center of distortion.
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        
        // Image principal points
        T px = T(cx);
        T py = T(cy);
        
        // Apply second and fourth order radial distortion.
        const T& l1 = camera[7];
        const T& l2 = camera[8];
        T r2 = xp*xp + yp*yp;
        T distortion = T(1.0) + r2  * (l1 + l2  * r2);
        
        // Compute final projected point position.
        const T& focal = camera[6];
        T predicted_x = (focal * distortion * xp) + px;
        T predicted_y = (focal * distortion * yp) + py;
        //T predicted_x = (focal * xp) + px;
        //T predicted_y = (focal * yp) + py;

        // The error is the squared euclidian distance between the predicted and observed position.
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);

        residuals[2] = plane[0]*point[0] + plane[1]*point[1] + plane[2]*point[2] + plane[3];
        residuals[2] /= sqrt( pow(plane[0],2) + pow(plane[1],2) + pow(plane[2],2) );
        residuals[2] /= T(num_cams);
        
        return true;
    }
    
    double observed_x;
    double observed_y;
    double cx;
    double cy;
    int num_cams;
    
};

// Pinhole Reprojection Error function
struct pinholeReprojectionError2 {
pinholeReprojectionError2(double observed_x, double observed_y, double cx, double cy, int num_cams, double* rvecs)
: observed_x(observed_x), observed_y(observed_y), cx(cx), cy(cy), num_cams(num_cams), rvecs(rvecs) {}
    template <typename T>
    bool operator()(const T* const camera_rest,
                    const T* const point,
                    const T* const plane,
                    T* residuals) const {
        
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        T r[3];
        for (int i=0; i<3; i++) {
            r[i] = T(rvecs[i]);
        }
        ceres::AngleAxisRotatePoint(r, point, p);
        
        // camera[3,4,5] are the translation.
        p[0] += camera_rest[0];
        p[1] += camera_rest[1];
        p[2] += camera_rest[2];
        
        // Compute the center of distortion.
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        
        // Image principal points
        T px = T(cx);
        T py = T(cy);
        
        // Apply second and fourth order radial distortion.
        const T& l1 = camera_rest[4];
        const T& l2 = camera_rest[5];
        T r2 = xp*xp + yp*yp;
        T distortion = T(1.0) + r2  * (l1 + l2  * r2);
        
        // Compute final projected point position.
        const T& focal = camera_rest[3];
        //T predicted_x = (focal * distortion * xp) + px;
        //T predicted_y = (focal * distortion * yp) + py;
        T predicted_x = (focal * xp) + px;
        T predicted_y = (focal * yp) + py;

        // The error is the squared euclidian distance between the predicted and observed position.
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);

        residuals[2] = plane[0]*point[0] + plane[1]*point[1] + plane[2]*point[2] + plane[3];
        residuals[2] /= sqrt( pow(plane[0],2) + pow(plane[1],2) + pow(plane[2],2) );
        residuals[2] /= T(num_cams);
        
        return true;
    }
    
    double observed_x;
    double observed_y;
    double cx;
    double cy;
    int num_cams;
    double* rvecs;
    
};

// Pinhole Reprojection Error function keeping R constant
struct alignmentError {
alignmentError(double* kr, int num_cams)
: kr(kr), num_cams(num_cams) {}
    template <typename T>
    bool operator()(const T* const params_all,
                    T* residuals) const {
        
        // Calculate dynamic aperture center parameters
        T center[3];
        for (int i=0; i<3; i++) {
            center[i] = T(0);
        }
        for (int i=0; i<num_cams; i++) {
            for (int j=0; j<3; j++) {
                center[j] += params_all[(i*6)+j]/double(num_cams);
            }
        }

        // Calculate current distance from origin for all cameras
        T d[num_cams];
        for (int i=0; i<num_cams; i++) {
            d[i] = T(0);
            for (int j=0; j<3; j++) {
                d[i] += pow(params_all[(i*6)+j+3],2);
            }
            d[i] = sqrt(d[i]);
        }

        // Calculate difference in relative pose between camera and
        // aperture center
        for (int i=1; i<num_cams; i++) {
            for (int j=0; j<3; j++) {
                residuals[((i-1)*3)+j] = kr[((i-1)*3)+j] - params_all[j] + params_all[(i*6)+j];
            }
        }

        // R and x and y params of t for aperture center as residuals
        for (int i=0; i<3; i++) {
            residuals[(3*(num_cams-1))+i] = center[i];
        }
        
        return true;
    }
    
    double* kr;
    double* kt;
    int num_cams;
    
};

// Read a Bundle Adjustment refractive dataset
class baProblem_ref {
 public:
    ~baProblem_ref() {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
    }
    
    int num_observations()       const { return num_observations_;               }
    const double* observations() const { return observations_;                   }
    double* mutable_cameras()          { return parameters_;                     }
    double* mutable_points()           { return parameters_  + 9 * num_cameras_; }
    double* mutable_scene_params()     { return parameters_ + 9 * num_cameras_ + 3 * num_points_; }
    
    double* mutable_camera_for_observation(int i) {
        return mutable_cameras() + camera_index_[i] * 9;
    }
    double* mutable_point_for_observation(int i) {
        return mutable_points() + point_index_[i] * 3;
    }
    
    bool LoadFile(const char* filename) {
        FILE* fptr = fopen(filename, "r");
        if (fptr == NULL) {
            return false;
        };
        
        FscanfOrDie(fptr, "%d", &num_cameras_);
        FscanfOrDie(fptr, "%d", &num_points_);
        FscanfOrDie(fptr, "%d", &num_observations_);
        
        point_index_ = new int[num_observations_];
        camera_index_ = new int[num_observations_];
        observations_ = new double[2 * num_observations_];
        
        num_parameters_ = 9 * num_cameras_ + 3 * num_points_ + 5;
        parameters_ = new double[num_parameters_];
        
        for (int i = 0; i < num_observations_; ++i) {
            FscanfOrDie(fptr, "%d", camera_index_ + i);
            FscanfOrDie(fptr, "%d", point_index_ + i);
            for (int j = 0; j < 2; ++j) {
                FscanfOrDie(fptr, "%lf", observations_ + 2*i + j);
            }
        }
        
        for (int i = 0; i < num_parameters_; ++i) {
            FscanfOrDie(fptr, "%lf", parameters_ + i);
        }
        return true;
    }
    
    int num_cameras_;
    int num_points_;

    double cx;
    double cy;
    
 private:
    template<typename T>
        void FscanfOrDie(FILE *fptr, const char *format, T *value) {
        int num_scanned = fscanf(fptr, format, value);
        if (num_scanned != 1) {
            LOG(FATAL) << "Invalid UW data file.";
        }
    }
    
    int num_observations_;
    int num_parameters_;
    
    int* point_index_;
    int* camera_index_;
    double* observations_;
    double* parameters_;
};

// Refractive Reprojection Error function
struct refractiveReprojectionError {
refractiveReprojectionError(double observed_x, double observed_y)
: observed_x(observed_x), observed_y(observed_y) {}
    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    const T* const scene,
                    T* residuals) const {
        
        
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        
        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        
        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        // change to supply from code
        T cx = T(646);
        T cy = T(482);
        
        // Apply second and fourth order radial distortion.
        const T& l1 = camera[7];
        const T& l2 = camera[8];
        T r2 = xp*xp + yp*yp;
        T distortion = T(1.0) + r2  * (l1 + l2  * r2);
        
        // Compute final projected point position.
        const T& focal = camera[6];
        T predicted_x = (focal * distortion * xp) + cx;
        T predicted_y = (focal * distortion * yp) + cy;
        //T predicted_x = (focal * xp) + cx;
        //T predicted_y = (focal * yp) + cy;


        // The error is the squared euclidian distance between the predicted and observed position.
        residuals[0] = pow((predicted_x - T(observed_x)),2) + pow((predicted_y - T(observed_y)),2);
        
        return true;
    }
    
    double observed_x;
    double observed_y;
    
};

class leastSquares {
 public:
    ~leastSquares() {
        delete[] parameters_;
    }

    double* mutable_params()          { return parameters_;                     }
    int num_parameters()              { return num_parameters_;                 }
    
    bool LoadFile(const char* filename) {
        FILE* fptr = fopen(filename, "r");
        if (fptr == NULL) {
            return false;
        };
        
        FscanfOrDie(fptr, "%d", &num_parameters_);
        
        parameters_ = new double[num_parameters_];
        
        for (int i = 0; i < num_parameters_; ++i) {
            FscanfOrDie(fptr, "%lf", parameters_ + i);
        }
        return true;
    }
    
 private:
    template<typename T>
        void FscanfOrDie(FILE *fptr, const char *format, T *value) {
        int num_scanned = fscanf(fptr, format, value);
        if (num_scanned != 1) {
            LOG(FATAL) << "Invalid UW data file.";
        }
    }
    
    int num_parameters_;   
    double* parameters_;
};

// Rotation Error function
struct rotationError {
rotationError(double x1, double y1, double z1, double x2, double y2, double z2)
: x1(x1), y1(y1), z1(z1), x2(x2), y2(y2), z2(z2) {}

    template <typename T>
    bool operator()(const T* const params,
                    T* residuals) const {
        
        // Applying rotation matrix
        T r[3];
        for (int i=0; i<3; i++) {
            r[i] = params[(i*3)+0]*T(x2)+params[(i*3)+1]*T(y2)+params[(i*3)+2]*T(z2);
        }

        // The error is the euclidian distance between points
        residuals[0] = pow(T(x2)-r[0], 2) + pow(T(y2)-r[1], 2) + pow(T(z2)-r[2], 2);
        
        return true;
    }
    
    double x1, y1, z1, x2, y2, z2;
    
};

// Plane Error function
struct planeError {
planeError(double x, double y, double z)
: x(x), y(y), z(z) {}

    template <typename T>
    bool operator()(const T* const params,
                    T* residuals) const {

        // The error is normal distance from plane
        residuals[0] = params[0]*T(x) + params[1]*T(y) + params[2]*T(z) + params[3];
        residuals[0] /= sqrt(pow(params[0], 2) + pow(params[1], 2) + pow(params[2], 2));
        
        T den = sqrt(pow(params[0], 2) + pow(params[1], 2) + pow(params[2], 2));

        // The error is square of deviation from plane equation
        residuals[0] = (params[0]*T(x) + params[1]*T(y) + params[2]*T(z) + params[3])/den;
        
        return true;
    }
    
    double x, y, z;
    
};



// FUNCTION DEFINITIONS

double BA_pinhole(baProblem &ba_problem, string ba_file, Size img_size);

double BA_pinhole2(baProblem &ba_problem, string ba_file, Size img_size, char* argv);

double BA_align(alignProblem &align_problem, string align_file);

double fit_plane(leastSquares &ls_problem, string filename, vector<Mat> points);

double BA_pinhole(baProblem &ba_problem, string ba_file, Size img_size, char* argv);

double fit_plane(leastSquares &ls_problem, string filename, char* argv, vector<Mat> points);

#endif
