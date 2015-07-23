#include "openfv.h"


int main(int argc, char* argv[]) {
     // Initialize Google's logging library.
     google::InitGoogleLogging(argv[0]);
	
     multiCamCalibration calibcam("/home/loganford16/Desktop/multiCamCalibration_Sample/data/pinhole1/", 6,5 , 5, 0, 0, 0, 0, 0);
     calibcam.run();

return 0;
}
