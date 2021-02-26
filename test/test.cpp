#include <glog/logging.h>
#include <gflags/gflags.h>

LOG(INFO)<<"It works.";
LOG(WARN)<<"Something not ideal";
LOG(ERROR)<<"Something went wrong";
LOG(FATAL)<<"AAAAAAAAAAAAAAA!";
VLOG(1)<<"one";
VLOG(2)<<"two";
VLOG(3)<<"three";
VLOG(4)<<"four";