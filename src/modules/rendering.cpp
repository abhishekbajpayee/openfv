// -------------------------------------------------------
// -------------------------------------------------------
// Synthetic Aperture - Particle Tracking Velocimetry Code
// --- Rendering Library ---
// -------------------------------------------------------
// Author: Abhishek Bajpayee
//         Dept. of Mechanical Engineering
//         Massachusetts Institute of Technology
// -------------------------------------------------------
// -------------------------------------------------------

#include "std_include.h"
#include "tools.h"
#include "rendering.h"

// #include <Eigen/Core>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace std;
using namespace cv;

// Scene class functions

Scene::Scene() {}

void Scene::create(double sx, double sy, double sz) {

    LOG(INFO)<<"Creating scene...";

    xlims_.push_back(-0.5*sx); xlims_.push_back(0.5*sx);
    ylims_.push_back(-0.5*sy); ylims_.push_back(0.5*sy);
    zlims_.push_back(-0.5*sz); zlims_.push_back(0.5*sz);

    sx_ = sx; sy_ = sy; sz_ = sz;
    
    
    REF_FLAG = 0;

    // TODO: this needs to be tweaked
    sigmax_ = 0.1;
    sigmay_ = 0.1;
    sigmaz_ = 0.1;

}

void Scene::setParticleSigma(double sx, double sy, double sz) {

    sigmax_ = sx;
    sigmay_ = sy;
    sigmaz_ = sz;
    
}

void Scene::setRefractiveGeom(float zW, float n1, float n2, float n3, float t) {

    REF_FLAG = 1;

    geom_.push_back(zW);
    geom_.push_back(n1); geom_.push_back(n2); geom_.push_back(n3);
    geom_.push_back(t);

}

void Scene::seedR() {

    LOG(INFO)<<"Seeding R...";

    particles_ = (Mat_<double>(4,13) << 0, 0,   0,   0,   0,   10,  20,  20,  20,  10,  0,   10,  20,
                                        0, -10, -20, -30, -40, -40, -40, -30, -20, -20, -20, -10, 0,
                                        0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                        1, 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1);

}

void Scene::seedParticles(vector< vector<double> > points) {

    LOG(INFO)<<"Seeding particles...";

    int num = points.size();
    particles_ = Mat_<double>::zeros(4, num);
    
    for (int i=0; i<num; i++) {
        particles_(0,i) = points[i][0]; particles_(1,i) = points[i][1]; particles_(2,i) = points[i][2]; particles_(3,i) = 1;
    }

}

void Scene::seedParticles(int num) {

    LOG(INFO)<<"Seeding particles...";

    particles_ = Mat_<double>::zeros(4, num);

    int res = 1000;

    double x, y, z;
    for (int i=0; i<num; i++) {

        x = (double(rand()%res)/double(res))*sx_ - 0.5*sx_;
        y = (double(rand()%res)/double(res))*sy_ - 0.5*sy_;
        z = (double(rand()%res)/double(res))*sz_ - 0.5*sz_;

        particles_(0,i) = x; particles_(1,i) = y; particles_(2,i) = z; particles_(3,i) = 1;

    }

}

void Scene::renderVolume(int xv, int yv, int zv) {

    vx_ = xv; vy_ = yv; vz_ = zv;

    voxelsX_ = linspace(-0.5*sx_, 0.5*sx_, vx_);
    voxelsY_ = linspace(-0.5*sy_, 0.5*sy_, vy_);
    voxelsZ_ = linspace(-0.5*sz_, 0.5*sz_, vz_);

    LOG(INFO)<<"Generating voxels...";

    double thresh = 0.1;

    for (int k=0; k<voxelsZ_.size(); k++) {
        LOG(INFO)<<k;
        for (int i=0; i<voxelsX_.size(); i++) {
            for (int j=0; j<voxelsY_.size(); j++) {
            
                double intensity = f(voxelsX_[i], voxelsY_[j], voxelsZ_[k]);
                if (intensity > thresh) {
                    voxel v;
                    v.x = i; v.y = j; v.z = k; v.I = int(intensity);
                    volume_.push_back(v);
                }

            }
        }
    }

    VLOG(1)<<"Seeded voxels: "<<volume_.size();

}

void Scene::renderVolumeGPU(int xv, int yv, int zv) {

    LOG(INFO)<<"GPU rendering voxels...";

    vx_ = xv; vy_ = yv; vz_ = zv;

    voxelsX_ = linspace(-0.5*sx_, 0.5*sx_, vx_);
    voxelsY_ = linspace(-0.5*sy_, 0.5*sy_, vy_);
    voxelsZ_ = linspace(-0.5*sz_, 0.5*sz_, vz_);

    Mat x = Mat::zeros(vy_, vx_, CV_32F);
    Mat y = Mat::zeros(vy_, vx_, CV_32F);
    Mat blank = Mat::zeros(vy_, vx_, CV_32F);

    // filling temp matrices
    tmp1.upload(x); tmp2.upload(x);
    slice.upload(x);

    for (int i=0; i<vy_; i++) {
        for (int j=0; j<vx_; j++) {
            x.at<float>(i,j) = voxelsX_[j];
            y.at<float>(i,j) = voxelsY_[i];
        }
    }
    
    gx.upload(x); gy.upload(y);

    vol.clear();

    for (int z=0; z<vz_; z++) {
        for (int k=0; k<particles_.cols; k++) {
        
            tmp2 = 0;

            // outputs -(x-ux)^2/2sig^2
            gpu::add(gx, Scalar(-particles_(0,k)), tmp1);
            gpu::pow(tmp1, 2.0, tmp1);
            gpu::multiply(tmp1, Scalar(-1.0/(2*sigmax_*sigmax_)), tmp1);
            gpu::add(tmp2, tmp1, tmp2);

            // outputs -(y-uy)^2/2sig^2
            gpu::add(gy, Scalar(-particles_(1,k)), tmp1);
            gpu::pow(tmp1, 2.0, tmp1);
            gpu::multiply(tmp1, Scalar(-1.0/(2*sigmay_*sigmay_)), tmp1);
            gpu::add(tmp2, tmp1, tmp2);

            gpu::add(tmp2, Scalar(-1.0*pow(voxelsZ_[z]-particles_(2,k),2.0)/(2*sigmaz_*sigmaz_)), tmp2);
        
            gpu::exp(tmp2, tmp2);
            gpu::add(slice, tmp2, slice);
        
        }
    
        Mat result(slice);
        vol.push_back(result.clone());
        //qimshow(result);

    }

}

double Scene::f(double x, double y, double z) {

    double intensity=0;
    double d, dx, dy, dz, b;

    for (int i=0; i<particles_.cols; i++) {
        // 0,25 for z part to increase effective sigma in z
        dx = pow(x-particles_(0,i), 2); dy = pow(y-particles_(1,i), 2); dz = pow(z-particles_(2,i), 2);
        d = dx + dy + dz;
        if (d<25)
        {
            b = exp( -dx/(2*pow(sigmax_, 2)) - dy/(2*pow(sigmay_, 2)) - dz/(2*pow(sigmaz_, 2)) ); // normalization factor?
            intensity += b;
        }
    }
    
    return(intensity*255.0);

}

Mat Scene::getImg(int zv) {

    Mat A = Mat::zeros(vy_, vx_, CV_8U);
    vector<voxel> slice = getVoxels(zv);

    for (int i=0; i<slice.size(); i++) {
        A.at<char>(slice[i].y, slice[i].x) = slice[i].I;
    }
    
    return(A.clone());

}

Mat Scene::getSlice(int zv) {

    return(vol[zv]);

}

vector<voxel> Scene::getVoxels(int z) {

    vector<voxel> slice;

    for (int i=0; i<volume_.size(); i++) {
        if (volume_[i].z == z)
            slice.push_back(volume_[i]);
    }
    
    return(slice);

}

Mat Scene::getParticles() {
    return(particles_);
}

vector<float> Scene::getRefGeom() {
    return(geom_);
}

int Scene::getRefFlag() {
    return(REF_FLAG);
}

double Scene::sigma() {
    return(sigmax_);
}

// Camera class functions

Camera::Camera() {

    K_ = Mat_<double>::zeros(3,3);
    R_ = Mat_<double>::zeros(3,3);
    C_ = Mat_<double>::zeros(3,1);
    t_ = Mat_<double>::zeros(3,1);

}

void Camera::init(double f, int imsx, int imsy) {

    LOG(INFO)<<"Initializing camera...";

    f_ = f;
    imsx_ = imsx;
    imsy_ = imsy;
    cx_ = 0.5*imsx;
    cy_ = 0.5*imsy;

    // K matrix
    K_(0,0) = f_;  K_(1,1) = f_;
    K_(0,2) = cx_; K_(1,2) = cy_;
    K_(2,2) = 1;

}

void Camera::setLocation(double x, double y, double z) {

    C_(0,0) = x; C_(1,0) = y; C_(2,0) = z;
    pointAt(0, 0, 0);

}

// Generates rotation matrix according to the right, up, out
// convention and point the camera at C to point p
void Camera::pointAt(double x, double y, double z) {

    Mat_<double> p = (Mat_<double>(3,1) << x, y, z);
    Mat_<double> up = (Mat_<double>(3,1) << 0, 1, 0);

    Mat_<double> L = p - C_; L = normalize(L);
    Mat_<double> s = cross(L, up);
    Mat_<double> u = cross(s, L);
    
    for (int i=0; i<3; i++) {
        R_(0,i) = -s(i,0); 
        R_(1,i) = u(i,0);
        R_(2,i) = -L(i,0);
    }

    t_ = -R_*C_;

    P_ = K_*Rt();

}

void Camera::setScene(Scene scene) {

    scene_ = scene;
    REF_FLAG = scene_.getRefFlag();
    if (REF_FLAG)
        geom_ = scene.getRefGeom();

}

Mat Camera::render() {

    project();

    LOG(INFO)<<"Rendering image...";

    Mat img = Mat::zeros(imsy_, imsx_, CV_8U);

    for (int i=0; i<img.rows; i++) {
        for (int j=0; j<img.cols; j++) {
            img.at<char>(i,j) = int( f(double(j), double(i)) );
        }
    }

    return(img.clone());

}

Mat Camera::renderGPU() {

    project();

    LOG(INFO)<<"Rendering image...";

    Mat x = Mat::zeros(imsy_, imsx_, CV_32F);
    Mat y = Mat::zeros(imsy_, imsx_, CV_32F);
    Mat blank = Mat::zeros(imsy_, imsx_, CV_32F);

    tmp1.upload(x), tmp2.upload(x);
    img.upload(x);

    for (int i=0; i<imsy_; i++) {
        for (int j=0; j<imsx_; j++) {
            x.at<float>(i,j) = float(j);
            y.at<float>(i,j) = float(i);
        }
    }

    gx.upload(x); gy.upload(y);
    
    for (int k=0; k<p_.cols; k++) {
        
        tmp2 = 0;

        // outputs -(x-ux)^2/2sig^2
        gpu::add(gx, Scalar(-p_(0,k)), tmp1);
        gpu::pow(tmp1, 2.0, tmp1);
        gpu::multiply(tmp1, Scalar(-1.0/(2*s_(0,k)*s_(0,k))), tmp1);
        gpu::add(tmp2, tmp1, tmp2);

        // outputs -(y-uy)^2/2sig^2
        gpu::add(gy, Scalar(-p_(1,k)), tmp1);
        gpu::pow(tmp1, 2.0, tmp1);
        gpu::multiply(tmp1, Scalar(-1.0/(2*s_(0,k)*s_(0,k))), tmp1);
        gpu::add(tmp2, tmp1, tmp2);
        
        gpu::exp(tmp2, tmp2);
        gpu::add(img, tmp2, img);
        
    }
    
    Mat result(img);
    return(result.clone());

}

Mat Camera::getP() {
    return(P_.clone());
}

Mat Camera::getC() {
    return(C_.clone());
}

// TODO: Consider GPU version of this?
void Camera::project() {

    VLOG(1)<<"Projecting points...";

    // project the points + sigmas
    Mat_<double> particles = scene_.getParticles();

    p_ = Mat_<double>::zeros(2, particles.cols);
    s_ = Mat_<double>::zeros(1, particles.cols);

    Mat_<double> proj;
    if (REF_FLAG) {
        Mat_<double> Xout = Mat_<double>::zeros(4, particles.cols);
        img_refrac(C_, particles, Xout);
        proj = P_*Xout;
    } else {
        proj = P_*particles;
    }
    double d;

    // TODO: optimize, consider wrapping in Eigen Matrix containers
    for (int i=0; i<proj.cols; i++) {
        
        proj(0,i) /= proj(2,i);
        proj(1,i) /= proj(2,i);
        p_(0,i) = proj(0,i);
        p_(1,i) = proj(1,i);
        
        // TODO: correct this for refractive and decide if needed at all because of diffraction limited imaging
        d = sqrt( pow(C_(0,0)-particles(0,i), 2) + pow(C_(1,0)-particles(1,i), 2) + pow(C_(2,0)-particles(2,i), 2) );
        s_(0,i) = scene_.sigma()*f_/d;

    }

}

void Camera::img_refrac(Mat_<double> Xcam, Mat_<double> X, Mat_<double> &X_out) {

    float zW_ = geom_[0]; float n1_ = geom_[1]; float n2_ = geom_[2]; float n3_ = geom_[3]; float t_ = geom_[4];

    double c[3];
    for (int i=0; i<3; i++)
        c[i] = Xcam.at<double>(i,0);

    for (int n=0; n<X.cols; n++) {

        double a[3];
        double b[3];
        double point[3];
        for (int i=0; i<3; i++)
            point[i] = X(i,n);

        a[0] = c[0] + (point[0]-c[0])*(zW_-c[2])/(point[2]-c[2]);
        a[1] = c[1] + (point[1]-c[1])*(zW_-c[2])/(point[2]-c[2]);
        a[2] = zW_;
        b[0] = c[0] + (point[0]-c[0])*(t_+zW_-c[2])/(point[2]-c[2]);
        b[1] = c[1] + (point[1]-c[1])*(t_+zW_-c[2])/(point[2]-c[2]);
        b[2] = t_+zW_;
        
        double rp = sqrt( pow(point[0]-c[0],2) + pow(point[1]-c[1],2) );
        double dp = point[2]-b[2];
        double phi = atan2(point[1]-c[1],point[0]-c[0]);

        double ra = sqrt( pow(a[0]-c[0],2) + pow(a[1]-c[1],2) );
        double rb = sqrt( pow(b[0]-c[0],2) + pow(b[1]-c[1],2) );
        double da = a[2]-c[2];
        double db = b[2]-a[2];
        
        double f, g, dfdra, dfdrb, dgdra, dgdrb;
        
        // Newton Raphson loop to solve for Snell's law
        double tol=1E-8;

        for (int i=0; i<10; i++) {

            f = ( ra/sqrt(pow(ra,2)+pow(da,2)) ) - ( (n2_/n1_)*(rb-ra)/sqrt(pow(rb-ra,2)+pow(db,2)) );
            g = ( (rb-ra)/sqrt(pow(rb-ra,2)+pow(db,2)) ) - ( (n3_/n2_)*(rp-rb)/sqrt(pow(rp-rb,2)+pow(dp,2)) );
            
            dfdra = ( (1.0)/sqrt(pow(ra,2)+pow(da,2)) )
                - ( pow(ra,2)/pow(pow(ra,2)+pow(da,2),1.5) )
                + ( (n2_/n1_)/sqrt(pow(ra-rb,2)+pow(db,2)) )
                - ( (n2_/n1_)*(ra-rb)*(2*ra-2*rb)/(2*pow(pow(ra-rb,2)+pow(db,2),1.5)) );

            dfdrb = ( (n2_/n1_)*(ra-rb)*(2*ra-2*rb)/(2*pow(pow(ra-rb,2)+pow(db,2),1.5)) )
                - ( (n2_/n1_)/sqrt(pow(ra-rb,2)+pow(db,2)) );

            dgdra = ( (ra-rb)*(2*ra-2*rb)/(2*pow(pow(ra-rb,2)+pow(db,2),1.5)) )
                - ( (1.0)/sqrt(pow(ra-rb,2)+pow(db,2)) );

            dgdrb = ( (1.0)/sqrt(pow(ra-rb,2)+pow(db,2)) )
                + ( (n3_/n2_)/sqrt(pow(rb-rp,2)+pow(dp,2)) )
                - ( (ra-rb)*(2*ra-2*rb)/(2*pow(pow(ra-rb,2)+pow(db,2),1.5)) )
                - ( (n3_/n2_)*(rb-rp)*(2*rb-2*rp)/(2*pow(pow(rb-rp,2)+pow(dp,2),1.5)) );

            ra = ra - ( (f*dgdrb - g*dfdrb)/(dfdra*dgdrb - dfdrb*dgdra) );
            rb = rb - ( (g*dfdra - f*dgdra)/(dfdra*dgdrb - dfdrb*dgdra) );

        }

        a[0] = ra*cos(phi) + c[0];
        a[1] = ra*sin(phi) + c[1];

        X_out(0,n) = a[0];
        X_out(1,n) = a[1];
        X_out(2,n) = a[2];
        X_out(3,n) = 1.0;

    }

}

double Camera::f(double x, double y) {

    double intensity=0;

    for (int i=0; i<p_.cols; i++) {
        double d = pow(x-p_(0,i), 2) + pow(y-p_(1,i), 2); 
        if (d<25)
        {
            //double d = pow(x-p_(0,i), 2) + pow(y-p_(1,i), 2);
            double b = exp( -d/(2*pow(s_(0,i), 2)) ); // normalization factor?
            //LOG(INFO)<<b;
            intensity += b;
        }

    }
    
    return(intensity*255.0);

}

Mat Camera::Rt() {

    Mat_<double> Rt = Mat_<double>::zeros(3,4);

    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            Rt(i,j) = R_(i,j);
        }
        Rt(i,3) = t_(i,0);
    }

    return(Rt);

}

// Python wrapper
BOOST_PYTHON_MODULE(rendering) {

    using namespace boost::python;

    void (Scene::*sPx1)(vector< vector<double> >) = &Scene::seedParticles;
    void (Scene::*sPx2)(int)                      = &Scene::seedParticles;

    class_<Scene>("Scene")
        .def("create", &Scene::create)
        .def("seedR", &Scene::seedR)
        .def("seedParticles", sPx2)
        .def("setParticleSigma", &Scene::setParticleSigma)
        .def("renderVolume", &Scene::renderVolume)
        .def("renderVolumeGPU", &Scene::renderVolumeGPU)
        .def("getSlice", &Scene::getSlice)
        .def("getImg", &Scene::getImg)
    ;

    class_<Camera>("Camera")
        .def("init", &Camera::init)
        .def("setScene", &Camera::setScene)
        .def("setLocation", &Camera::setLocation)
        .def("render", &Camera::render)
        .def("renderGPU", &Camera::renderGPU)
        .def("getP", &Camera::getP)
        .def("getC", &Camera::getC)
    ;

}
