//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//                           License Agreement
//                For Open Source Flow Visualization Library
//
// Copyright 2013-2017 Abhishek Bajpayee
//
// This file is part of OpenFV.
//
// OpenFV is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License version 2 as published by the Free Software Foundation.
//
// OpenFV is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License version 2 for more details.
//
// You should have received a copy of the GNU General Public License version 2 along with
// OpenFV. If not, see https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html.

#include "std_include.h"

#include "refocusing.h"
#include "typedefs.h"

using namespace cv;
using namespace std;

boost::program_options::options_description get_options();

/*! Function to parse a config file and return a reconstruction_settings variable.
  \param filename Full path to configuration file
  \param settings Destination reconstruction_settings variable
  \param help Flag to show all options as a help menu instead of parsing data
*/
void parse_reconstruction_settings(string filename, reconstruction_settings &settings, bool help);

/*! Function to parse a refocus settings file and return a refocus_settings variable
  which can be directly passed to an saRefocus constructor.
  \param filename Full path to configuration file
  \param settings Destination refocus_settings variable
  \param help Flag to show all options as a help menu instead of parsing data
*/
void parse_refocus_settings(string filename, refocus_settings &settings, bool help);

/*! Function to parse a calibration settings file and return a calibration_settings variable
  which can be directly passed to an saRefocus constructor.
  \param filename Full path to configuration file
  \param settings Destination calibration_settings variable
  \param help Flag to show all options as a help menu instead of parsing data
void parse_calibration_settings(string filename, calibration_settings &settings, bool help);
*/
