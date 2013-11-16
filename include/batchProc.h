#ifndef BATCH_PROCESS_LIBRARY
#define BATCH_PROCESS_LIBRARY

#include "std_include.h"

#include "calibration.h"
#include "refocusing.h"
#include "pLoc.h"
#include "tracking.h"
#include "tools.h"
#include "optimization.h"
#include "typedefs.h"

class batchFind {

 public:

    ~batchFind() {

    }

    batchFind(string path);

    void run();
    void read_config_file();

 private:

    string path_;

    vector<string> calib_paths;
    vector<string> refoc_paths;
    vector<float> threshs;

    int n_;

};

#endif
