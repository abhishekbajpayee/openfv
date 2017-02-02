//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//                           License Agreement
//                For Open Source Flow Visualization Library
//
// Copyright 2013-2015 Abhishek Bajpayee
//
// This file is part of openFV.
//
// openFV is free software: you can redistribute it and/or modify it under the terms of the 
// GNU General Public License as published by the Free Software Foundation, either version 
// 3 of the License, or (at your option) any later version.
//
// openFV is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
// See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with openFV. 
// If not, see http://www.gnu.org/licenses/.

// -------------------------------------------------------
// -------------------------------------------------------
// Synthetic Aperture - Particle Tracking Velocimetry Code
// --- Refocusing Library Header ---
// -------------------------------------------------------
// Author: Abhishek Bajpayee
//         Dept. of Mechanical Engineering
//         Massachusetts Institute of Technology
// -------------------------------------------------------
// -------------------------------------------------------

// ----------------------
// Includes all of OpenFV
// ----------------------

// Core
#include "std_include.h"
#include "typedefs.h"
#include "calibration.h"
#include "optimization.h"
#include "refocusing.h"
#include "rendering.h"
#include "serialization.h"
#include "parse_settings.h"
#include "tools.h"

// Particle Localization and Tracking
#include "pLoc.h"
#include "tracking.h"

// CUDA
#pragma message "CUDA_FLAG="
#ifndef _WITHOUT_CUDA_
#include "cuda_lib.h"
#endif

// Other
#include "piv.h"
#include "visualize.h"
#include "batchProc.h"

// SAFE
#include "featureDetection.h"
#include "parse_safe_settings.h"
#include "safeRefocusing.h"
