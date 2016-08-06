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
#include "cuda_lib.h"

// Other
#include "piv.h"
#include "visualize.h"
#include "batchProc.h"

// SAFE
#include "featureDetection.h"
#include "parse_safe_settings.h"
#include "safeRefocusing.h"
