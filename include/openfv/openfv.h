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
#include "optimization.h"
#include "refocusing.h"
#include "rendering.h"
#include "serialization.h"
#include "parse_settings.h"
#include "tools.h"

// Particle Localization and Tracking
#include "tracking.h"

// CUDA
#ifndef _WITHOUT_CUDA_
#include "cuda_lib.h"
#endif

// LEGACY INCLUDES

// Core
// #include "calibration.h"

// Other
// #include "piv.h"
// #include "visualize.h"
// #include "batchProc.h"

// SAFE
// #include "featureDetection.h"
// #include "parse_safe_settings.h"
// #include "safeRefocusing.h"