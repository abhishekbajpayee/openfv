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

    baProblem() {
        point_index_ = NULL;
        camera_index_ = NULL;
        plane_index_ = NULL;
        observations_ = NULL;
        parameters_ = NULL;
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
    
};

// Read a refractive Bundle Adjustment dataset
class baProblem_ref {
 
 public:
    ~baProblem_ref() {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] plane_index_;
        delete[] observations_;
        delete[] parameters_;
    }

    baProblem_ref() {
        point_index_ = NULL;
        camera_index_ = NULL;
        plane_index_ = NULL;
        observations_ = NULL;
        parameters_ = NULL;
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

    double t()                         { return t_; }
    double n1()                        { return n1_; }
    double n2()                        { return n2_; }
    double n3()                        { return n3_; }
    
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
        
        FscanfOrDie(fptr, "%lf", &t_);
        FscanfOrDie(fptr, "%lf", &n1_);
        FscanfOrDie(fptr, "%lf", &n2_);
        FscanfOrDie(fptr, "%lf", &n3_);

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

    double t_, n1_, n2_, n3_;

};

// Refractive Reprojection Error function
class refractiveReprojectionError {

 public:

 refractiveReprojectionError(double observed_x, double observed_y, double cx, double cy, int num_cams, double t, double n1, double n2, double n3)
     : observed_x(observed_x), observed_y(observed_y), cx(cx), cy(cy), num_cams(num_cams), t_(t), n1_(n1), n2_(n2), n3_(n3) {}
    
    template <typename T>
        bool operator()(const T* const camera,
                        const T* const point,
                        const T* const plane,
                        T* residuals) const {
        
        // Inital guess for points on glass
        T* R = new T[9];
        ceres::AngleAxisToRotationMatrix(camera, R);

        T c[3];
        for (int i=0; i<3; i++) {
            c[i] = T(0);
            for (int j=0; j<3; j++) {
                c[i] += -R[i*1 + j*3]*camera[j+3];
            }
        }
        //cout<<"c: ("<<c[0]<<", "<<c[1]<<", "<<c[2]<<")\n";

        // All the refraction stuff
        T* a = new T[3];
        T* b = new T[3];
        a[0] = c[0] + (point[0]-c[0])*(T(-t_)-c[2])/(point[2]-c[2]);
        a[1] = c[1] + (point[1]-c[1])*(T(-t_)-c[2])/(point[2]-c[2]);
        a[2] = T(-t_);
        //cout<<"t "<<t_<<endl;
        //cout<<"n1 "<<n1_<<endl;
        //cout<<"a: ("<<a[0]<<", "<<a[1]<<", "<<a[2]<<")\n";
        b[0] = c[0] + (point[0]-c[0])*(-c[2])/(point[2]-c[2]);
        b[1] = c[1] + (point[1]-c[1])*(-c[2])/(point[2]-c[2]);
        b[2] = T(0);
        
        T rp = sqrt( pow(point[0]-c[0],2) + pow(point[1]-c[1],2) );
        T dp = point[2]-b[2];
        T phi = atan2(point[1]-c[1],point[0]-c[0]);

        T ra = sqrt( pow(a[0]-c[0],2) + pow(a[1]-c[1],2) );
        T rb = sqrt( pow(b[0]-c[0],2) + pow(b[1]-c[1],2) );
        T da = a[2]-c[2];
        T db = b[2]-a[2];

        T f, g, dfdra, dfdrb, dgdra, dgdrb;
        
        // Newton Raphson loop to solve for Snell's law
        for (int i=0; i<10; i++) {

            f = ( ra/sqrt(pow(ra,2)+pow(da,2)) ) - ( T(n2_/n1_)*(rb-ra)/sqrt(pow(rb-ra,2)+pow(db,2)) );
            g = ( (rb-ra)/sqrt(pow(rb-ra,2)+pow(db,2)) ) - ( T(n3_/n2_)*(rp-rb)/sqrt(pow(rp-rb,2)+pow(dp,2)) );
            
            dfdra = ( T(1.0)/sqrt(pow(ra,2)+pow(da,2)) )
                - ( pow(ra,2)/pow(pow(ra,2)+pow(da,2),1.5) )
                + ( T(n2_/n2_)/sqrt(pow(ra-rb,2)+pow(db,2)) )
                - ( T(n2_/n1_)*(ra-rb)*(T(2)*ra-T(2)*rb)/(T(2)*pow(pow(ra-rb,2)+pow(db,2),1.5)) );

            dfdrb = ( T(n2_/n1_)*(ra-rb)*(T(2)*ra-T(2)*rb)/(T(2)*pow(pow(ra-rb,2)+pow(db,2),1.5)) )
                - ( T(n2_/n2_)/sqrt(pow(ra-rb,2)+pow(db,2)) );

            dgdra = ( (ra-rb)*(T(2)*ra-T(2)*rb)/(T(2)*pow(pow(ra-rb,2)+pow(db,2),1.5)) )
                - ( T(1.0)/sqrt(pow(ra-rb,2)+pow(db,2)) );

            dgdrb = ( T(1.0)/sqrt(pow(ra-rb,2)+pow(db,2)) )
                + ( T(n3_/n2_)/sqrt(pow(rb-rp,2)+pow(dp,2)) )
                - ( (ra-rb)*(T(2)*ra-T(2)*rb)/(T(2)*pow(pow(ra-rb,2)+pow(db,2),1.5)) )
                - ( T(n3_/n2_)*(rb-rp)*(T(2)*rb-T(2)*rp)/(T(2)*pow(pow(rb-rp,2)+pow(dp,2),1.5)) );

            ra = ra - ( (f*dgdrb - g*dfdrb)/(dfdra*dgdrb - dfdrb*dgdra) );
            rb = rb - ( (g*dfdra - f*dgdra)/(dfdra*dgdrb - dfdrb*dgdra) );
            //cout<<"val: "<<ra<<endl;

        }
        
        a[0] = ra*cos(phi) + c[0];
        a[1] = ra*sin(phi) + c[1];
        //cout<<"a: ("<<a[0]<<", "<<a[1]<<", "<<a[2]<<")\n";
        
        // Continuing projecting point a to camera
        T p[3];
        //ceres::AngleAxisRotatePoint(camera, point, p);
        ceres::AngleAxisRotatePoint(camera, a, p);

        // camera[3,4,5] are the translation.
        p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];
        
        // Compute the center of distortion.
        T xp = p[0] / p[2]; T yp = p[1] / p[2];
        
        // Image principal points
        T px = T(cx); T py = T(cy);
        
        // Apply second and fourth order radial distortion.
        const T& l1 = camera[7];
        const T& l2 = camera[8];
        T r2 = xp*xp + yp*yp;
        T distortion = T(1.0) + r2  * (l1 + l2  * r2);
        
        // Compute final projected point position.
        const T& focal = camera[6];
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
    
    double observed_x, observed_y, cx, cy, t_, n1_, n2_, n3_;
    int num_cams;
    
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

double BA_pinhole(baProblem &ba_problem, string ba_file, Size img_size, vector<int> const_points);

double BA_refractive(baProblem_ref &ba_problem, string ba_file, Size img_size, vector<int> const_points);

double fit_plane(leastSquares &ls_problem, string filename, vector<Mat> points);

#endif
