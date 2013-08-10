#include "std_include.h"

#include "calibration.h"
#include "refocusing.h"
#include "pLoc.h"
#include "tracking.h"
#include "tools.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

    Mat_<double> rot = Mat_<double>::zeros(1,3);
    rot(0,0) = 0.148279; rot(0,1) = 0.149589, rot(0,2) = 0.0113288;

    Mat_<double> trans = Mat_<double>::zeros(3,1);
    trans(0,0) = -38.06; trans(1,0) = -3.953, trans(2,0) = 1021.13;

    Mat R;
    Rodrigues(rot, R);
    
    Mat cam = -R*trans;
    cout<<cam<<endl;
    double c[3];
    c[0] = cam.at<double>(0,0);
    c[1] = cam.at<double>(0,1);
    c[2] = cam.at<double>(0,2);
        
    double point[3];
    point[0] = 16; point[1] = 22; point[2] = 31;

    double t, n1_, n2_, n3_;
    t = 5.0; n1_ = 1.0; n2_ = 1.517; n3_ = 1.0;

    // All the refraction stuff
    double a[3], b[3];
    a[0] = c[0] + (point[0]-c[0])*(-t-c[2])/(point[2]-c[2]);
    a[1] = c[1] + (point[1]-c[1])*(-t-c[2])/(point[2]-c[2]);
    a[2] = -t;
    b[0] = c[0] + (point[0]-c[0])*(-c[2])/(point[2]-c[2]);
    b[1] = c[1] + (point[1]-c[1])*(-c[2])/(point[2]-c[2]);
    b[2] = 0;

    double ra, rb, rp, da, db, dp, phi;

    rp = sqrt( pow(point[0]-c[0],2) + pow(point[1]-c[1],2) );
    dp = point[2]-b[2];
    phi = atan2(point[1]-c[1],point[0]-c[0]);

    ra = sqrt( pow(a[0]-c[0],2) + pow(a[1]-c[1],2) );
    rb = sqrt( pow(b[0]-c[0],2) + pow(b[1]-c[1],2) );
    da = a[2]-c[2];
    db = b[2]-a[2];
    
    double f, g, dfdra, dfdrb, dgdra, dgdrb;
    cout<<da<<" "<<db<<" "<<dp<<endl;
    // Newton Raphson loop to solve for Snell's law
    for (int i=0; i<10; i++) {
        
        f = ( ra/sqrt(pow(ra,2)+pow(da,2)) ) - ( (n2_/n1_)*(rb-ra)/sqrt(pow(rb-ra,2)+pow(db,2)) );
        g = ( (rb-ra)/sqrt(pow(rb-ra,2)+pow(db,2)) ) - ( (n3_/n2_)*(rp-rb)/sqrt(pow(rp-rb,2)+pow(dp,2)) );
        
        dfdra = ( 1.0/sqrt(pow(ra,2)+pow(da,2)) )
            - ( pow(ra,2)/pow(pow(ra,2)+pow(da,2),1.5) )
            + ( (n2_/n2_)/sqrt(pow(ra-rb,2)+pow(db,2)) )
            - ( (n2_/n1_)*(ra-rb)*(2*ra-2*rb)/(2*pow(pow(ra-rb,2)+pow(db,2),1.5)) );
        
        dfdrb = ( (n2_/n1_)*(ra-rb)*(2*ra-2*rb)/(2*pow(pow(ra-rb,2)+pow(db,2),1.5)) )
            - ( (n2_/n2_)/sqrt(pow(ra-rb,2)+pow(db,2)) );
        
        dgdra = ( (ra-rb)*(2*ra-2*rb)/(2*pow(pow(ra-rb,2)+pow(db,2),1.5)) )
            - ( 1.0/sqrt(pow(ra-rb,2)+pow(db,2)) );
        
        dgdrb = ( 1.0/sqrt(pow(ra-rb,2)+pow(db,2)) )
            + ( (n3_/n2_)/sqrt(pow(rb-rp,2)+pow(dp,2)) )
            - ( (ra-rb)*(2*ra-2*rb)/(2*pow(pow(ra-rb,2)+pow(db,2),1.5)) )
            - ( (n3_/n2_)*(rb-rp)*(2*rb-2*rp)/(2*pow(pow(rb-rp,2)+pow(dp,2),1.5)) );
        
        ra = ra - ( (f*dgdrb - g*dfdrb)/(dfdra*dgdrb - dfdrb*dgdra) );
        rb = rb - ( (g*dfdra - f*dgdra)/(dfdra*dgdrb - dfdrb*dgdra) );

        cout<<f<<endl;

    }
    
    a[0] = ra*cos(phi) + c[0];
    a[1] = ra*sin(phi) + c[1];

    cout<<"a "<<a[0]<<" "<<a[1]<<" "<<a[2]<<endl;

    return 1;

}
