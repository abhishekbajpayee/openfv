function [tomoOpts] = defineTomoOptions(varargin)
%DEFINETOMOOPTIONS Create an options structure for reconstruction
%
%   This allows the options used in the various reconstruction algorithms
%   to be defined before execution, saved, and/or passed around different
%   functions and scripts. Makes an easier interface for the reconstruction
%   functions and will ultimately allow a single harness for all
%   reconstruction types.
%
% Syntax:  
%       [tomoOpts] = defineTomoOptions('Parameter', Value,...)
%       	Returns an options structure specifying the settings used to 
%        	perform tomographic reconstruction.
%
% Parameters:
%
%       weighting       string      The weighting type to use. Default 'circle'.
%                                   See weightfunction.m
%
%       pixelThreshMart [1 x 1]     The intensity threshold below which 
%                                   pixels are deemed to have a zero value.
%                                   The higher this is, the more sparse
%                                   (and therefore quick to solve) the
%                                   problem is.
%
%       pixelThreshWrs  [1 x 1]     As per pixelThreshMart but tends to be
%                                   higher to reduce the size of the wrs
%                                   matrix. In the event of segmentation
%                                   faults, the most likely case is this
%                                   value being too low.
%
%       pixelThreshMfg  [1 x 1]     As per pixelThreshMart but invoked for
%                                   the MFG-MART algorithm.
%
%       voxelThreshMart [1 x 1]     The intensity threshold below which a 
%                                   voxel is deemed to have a zero value.
%                                   The higher this is, the more sparse
%                                   (and therefore quick to solve) the
%                                   problem is.
%
%       voxelThreshMfg  [1 x 1]     As per voxelThreshMart for wthe
%                                   MFG-MART algorithm
%
%       muMart          [1 x 1]     Relaxation parameter for MART 
%                                   reconstruction.
%
%       muMfg           [1 x 1]     Relaxation parameter for MFG-MART 
%                                   reconstruction.
%
%       nMartIters      [1 x 1]     Number of Mart iterations
%
%       nWrsIters       [1 x 1]     Number of Wrs solution iterations (when
%                                   using the simplematlab or
%                                   preconditionedmatlab algorithms only.
%                                   See help optimset() for solver control
%                                   of lsqlin and lsqnonneg).
%
%       nMfgIters       [1 x 1]     Number of Mart iterations following the
%                                   application of MFG. Can be 0!
%
%       muSmart         [1 x 1]     If 'smart' is used as a wrs
%                                   postprocessing algorithm (see
%                                   'methodWrs') this is the relaxaton
%                                   parameter used.
%
%       methodWrs       string      Defines the method used to solve the
%                                   Wrs matrix; alternatives are
%                                   'simplematlab' - an iterative curvature
%                                                    maximisation routine
%                                                    coded in MATLAB
%                                   'preconditionedmatlab'
%                                                  - a preconditioned
%                                                    iterative curvature
%                                                    maximisation routine
%                                                    coded in MATLAB.
%                                                    Generally not
%                                                    effective as the
%                                                    matrix is not
%                                                    diagonally dominant
%                                   'lsqlin'       - Uses the lsqlin
%                                                    function from MATLAB's
%                                                    optimization toolbox
%                                                    with a lower bound of
%                                                    zero on the solution
%                                   'lsqnonneg'    - Uses the NNLS
%                                                    algorithm, embedded
%                                                    into MATLAB. Do not
%                                                    use with MATLAB R2008a
%                                                    and lower as the
%                                                    matrix is converted to
%                                                    full.
%
%       solverOptions   struct      Solver options structure for use with
%                                   the lsqlin and lsqnonneg options (wrs
%                                   algorithm only). This should be the
%                                   output of MATLAB'S optimset() function.
%
%       algorithm       string      Details which reconstruction type to
%                                   use. Valid options are:
%                                       'wrs'
%                                       'mart'
%                                       'mart_pvr'
%                                       'mfg_mart'
%                                       'mart_parbatch'
%                                   
%
% References:               none
% Other m-files required:   none
% Subfunctions:             none
% Nested functions:         none
% MAT-files required:       none
%
% Author:                   T.H. Clark
% Work address:             Fluids Lab
%                           Cambridge University Engineering Department
%                           2 Trumpington Street
%                           Cambridge
%                           CB21PZ
% Email:                    t.clark@cantab.net
% Website:                  http://cambridge.academia.edu/ThomasClark/
%
% Revision History:        	01 August 2011      Created (based on
%                                               definePIVOptions.m)

%   Copyright (c) 2007-2015  Thomas H. Clark
% Algorithm to use. Alternatives are:
%       'wrs'
%       'mart'
%       'mart_pvr'
%       'mfg_mart'
%       'mart_parbatch'
tomoOpts.algorithm	= 'mart';

% Weighting type to use ('circle' or 'gaussian')
tomoOpts.weighting = 'circle';

% Options for WRS reconstruction
tomoOpts.solverOptions      = []; % lsqlin and lsqnonneg only
tomoOpts.nWrsIters          = 200; % simplematlab and preconditionedmatlab only
tomoOpts.pixelThreshWrs     = 0.06;
tomoOpts.methodWrs          = 'simple'; % alternatives are 'simple', 'preconditioned', 'lsqlin', 'lsqnonneg', 'backslash', 'smart'
tomoOpts.muSmart            = 1;

% Options for MART reconstruction
tomoOpts.nMartIters         = 5;
tomoOpts.muMart             = 0.4;
tomoOpts.pixelThreshMart    = 2/255;
tomoOpts.voxelThreshMart    = 100*eps('single');

% Options for MFG-MART reconstruction
tomoOpts.nMfgIters          = 3;
tomoOpts.muMfg              = 0.9;
tomoOpts.pixelThreshMfg     = 2/255;
tomoOpts.voxelThreshMfg     = 100*eps('single');

% Option to save the reconstruction file
tomoOpts.storeResult	    = 0; % numeric prevents save. If a string, this should be a full path and file name to an output *.mat file, including extension.

% Parse variable arguments (containing non-default values) into opts structure
tomoOpts = parse_pv_pairs(tomoOpts, varargin);


