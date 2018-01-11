// Author: Sudeep Pillai (spillai@csail.mit.edu)
// License: BSD
// Last modified: Sep 14, 2014

// Copyright (c) 2014, Sudeep Pillai
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.

// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "numpy-opencv-converter/np_opencv_converter.hpp"

namespace fs { namespace python {

// Static PyInit
static void py_init() {
  Py_Initialize();
  import_array();
}

// Singleton init and export converters
static bool export_type_conversions_once = false;
bool init_and_export_converters() {

  if (export_type_conversions_once)
    return false;
  
  std::cerr << "PYTHON TYPE CONVERTERS exported" << std::endl;
  export_type_conversions_once = true;

  // Py_Init and array import
  py_init();
  
  // => py::list
  expose_template_type<int>();
  expose_template_type<float>();
  expose_template_type<double>();

  // std::vector => py::list
  expose_template_type< std::vector<int> >();
  expose_template_type< std::vector<float> >();
  expose_template_type< std::vector<double> >();

  expose_template_type< std::vector<cv::Point> >();
  expose_template_type< std::vector<cv::Point2f> >();
  expose_template_type< std::vector<cv::KeyPoint> >();
  
  expose_template_type< std::vector<cv::Mat> >();
  expose_template_type< std::vector<cv::Mat1b > >();
  expose_template_type< std::vector<cv::Mat1f > >();

  // std::map => py::dict
  expose_template_type<std::map<int, std::vector<int> > >();
  expose_template_type<std::map<int, std::vector<float> > >();
  expose_template_type<std::map<std::string, float> >();

  // various converters to cv::Mat
  py::to_python_converter<cv::Point, Point_to_mat>();
  py::to_python_converter<cv::Point2f, Point2f_to_mat>();
  py::to_python_converter<cv::Point3f, Point3f_to_mat>();
  py::to_python_converter<cv::Vec3f, Vec3f_to_mat>();

  // register the to-from-python converter for each of the types
  Mat_PyObject_converter< cv::Mat >();
  
  // 1-channel
  Mat_PyObject_converter< cv::Mat1b >();
  Mat_PyObject_converter< cv::Mat1s >();
  Mat_PyObject_converter< cv::Mat1w >();
  Mat_PyObject_converter< cv::Mat1i >();
  Mat_PyObject_converter< cv::Mat1f >();
  Mat_PyObject_converter< cv::Mat1d >();

  // 2-channel
  Mat_PyObject_converter< cv::Mat2b >();
  Mat_PyObject_converter< cv::Mat2s >();
  Mat_PyObject_converter< cv::Mat2w >();
  Mat_PyObject_converter< cv::Mat2i >();
  Mat_PyObject_converter< cv::Mat2f >();
  Mat_PyObject_converter< cv::Mat2d >();

  // 3-channel
  Mat_PyObject_converter< cv::Mat3b >();
  Mat_PyObject_converter< cv::Mat3s >();
  Mat_PyObject_converter< cv::Mat3w >();
  Mat_PyObject_converter< cv::Mat3i >();
  Mat_PyObject_converter< cv::Mat3f >();
  Mat_PyObject_converter< cv::Mat3d >();

  // add more if needed

  return true;
}

} // namespace python
} // namespace fs

