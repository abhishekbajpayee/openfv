#include "std_include.h"

#include "safeRefocusing.h"

using namespace cv;
using namespace std;

void parse_safe_settings(string filename, safe_refocus_settings &safe_settings, bool h) {

  namespace po = boost::program_options;

  po::options_description desc("Allowed config file options");
  desc.add_options()
    ("dp", po::value<double>(), "Inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has half as big width and height.")
    ("minDist", po::value<double>(), "Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.")
    ("param1", po::value<double>(), "First method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the higher threshold of the two passed to the Canny() edge detector (the lower one is twice smaller).")
    ("param2", po::value<double>(), "Second method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.")
    ("minRadius", po::value<int>(), "Minimum circle radius.")
    ("maxRadius", po::value<int>(), "Maximum circle radius.")
    ("gKerWid", po::value<int>(), "Gaussian kernel width.")
    ("gKerHt", po::value<int>(), "Gaussian kernel height.")
    ("gKerSigX", po::value<int>(), "Gaussian kernel sigmaX.")
    ("gKerSigY", po::value<int>(), "Gaussian kernel sigmaY.")
    ("circle_rim_thickness", po::value<int>(), "Circle rim thickness for circle drawing on preprocessed images.")
    ("debug", po::value<int>(), "ON for SAFE debug mode");

  if (h) {
    cout<<desc;
    exit(1);
  }

  po::variables_map vm;
  po::store(po::parse_config_file<char>(filename.c_str(), desc), vm);
  po::notify(vm);

  safe_settings.dp = vm["dp"].as<double>();
  safe_settings.minDist = vm["minDist"].as<double>();
  safe_settings.param1 = vm["param1"].as<double>();
  safe_settings.param2 = vm["param2"].as<double>();	
  safe_settings.minRadius = vm["minRadius"].as<int>();
  safe_settings.maxRadius = vm["maxRadius"].as<int>();

  safe_settings.gKerWid = vm["gKerWid"].as<int>();
  safe_settings.gKerHt = vm["gKerHt"].as<int>();
  safe_settings.gKerSigX = vm["gKerSigX"].as<int>();
  safe_settings.gKerSigY = vm["gKerSigY"].as<int>();
  safe_settings.circle_rim_thickness = vm["circle_rim_thickness"].as<int>(); 
  safe_settings.debug = vm["debug"].as<int>();
}
