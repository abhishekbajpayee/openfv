function [pivOpts] = definePIVOptions(varargin)
%DEFINEPIVOPTIONS Create an options structure for PIV_3d
%
%   This allows the options used in the PIV_3d algorithm to be defined before
%   execution of the PIV algorithm, saved, and/or passed around different
%   functions and scripts.
%
% Syntax:  
%       [pivOpts] = definePIVOptions('Parameter', Value,...)
%       	Returns an options structure specifying the settings used to 
%        	perform PIV.
%
% References:
%   [1] Raffal M. Willert C. Wereley S. and Kompenhans J. 
%       'Particle Image Velocimetry (A Practical Guide)' 
%       2nd Ed., Springer,  ISBN 978-3-540-72307-3
% 
% Future Improvements:
%   [1] Input checking, mainly to make sure specified options are compatible and
%       valid, and especially to make sure that everything needing to be nPasses
%       in size is the same size.
%   
% Author:                   T. H. Clark
% Email:                    t.clark@cantab.net
%
% Revision History:        	08 July 2010        Created
%
%                           15 April 2011       Modified the algorithm option to
%                                               be consistent with the new
%                                               implementation of PIV_3d. This
%                                               alters the algorithm option from
%                                               being a function handle to being
%                                               a string specifying which
%                                               algorithm is called by PIV_3d()
%
%                           15 August 2011      Altered O2 to CTE
%
%                           02 November 2015    Updated to reflect new piv3d
%                                               code.
%
%   Copyright (c) 2007-2015  Thomas H. Clark


% DEFINE DEFAULT OPTIONS STRUCTURE:

% Options for the gap filling routine PIV_3d_gapfill
%   gapFillMethod may be 'linear' 'natural' or 'nearest'
pivOpts.gapFillMethod	= 'linear';

% Options for PIV_3d_maxdisplacement
%       eps_PIV_noise           scalar, typically 0.1-0.2 
%                               Mean noise level of the PIV data. This can be
%                               determined by plotting a histogram of intensity
%                               in the reconstructed volumes, and observing the
%                               amount of low-level noise.
%
%       eps_threshold           No idea what this value should take. 
%                               Threshold of median test acceptability, denoted
%                               as epsilon_thresh, p.185, section 6.1.5, ref[1].
pivOpts.maxUx           = Inf;
pivOpts.maxUy           = Inf;
pivOpts.maxUz           = Inf;
pivOpts.eps_PIV_noise	= [0.1 0.1 0.1];
pivOpts.eps_threshold	= [2 2 2];
pivOpts.maxSNR          = Inf;

% Options for limiting turbulent intensity.
% see help PIV_3d_turbulentintensity for more details
pivOpts.turbIntensity	= [Inf  Inf  Inf
                           Inf  Inf  Inf
                           Inf  Inf  Inf];
pivOpts.meanVelocity	= [NaN NaN NaN];


% Options for PIV_3d run parameters
pivOpts.nPasses         = 3;
pivOpts.wSize           = [64 64 64; 48 48 48; 32 32 32];
pivOpts.overlap         = [50 50 50];
pivOpts.edgeCut         = 4; % number of voxels to cut from the edge of each volume. NB this also cuts from the top and bottom. Must be at least 4 to use the fPIV algorithm
pivOpts.storeResult	    = 0; % numeric prevents save. If a string, this should be a full path and file name to an output *.mat file, including extension.
pivOpts.plot            = 0;
pivOpts.peakFinder      = 'gaussian'; % alternatives are 'gaussian' 'whittaker' and 'cubic'. 'cubic' sucks!
pivOpts.iwMaxDisp       = 50;

% Option to decide which PIV algorithm to use. NB deprecating a lot of old
% options for the purpose of tidying up
pivOpts.algorithm       = 'vodim3d';

% Option to add a first guess field.
% THIS MUST BE CONVERTED TO VOXELS/SECOND UNITS ALREADY!!!!
pivOpts.firstGuessField = NaN;

% Option to alter the interpolation routine used for fMexPIV_o2:
%   1 = direct interpolation (no convection)
%   2 = linear interpolation
%   3 = 5^3 cardinal interpolation
%   4 = 7^3 cardinal interpolation
pivOpts.fetchType = [1 2 3];

% Option to smooth the correlation plane (currently not implemented so default
% false
pivOpts.smoothCC = false; 

% Option to save the raw velocity field outputs (before vector field
% validation) to allow experimentation with different vector validation
% routines without going through the cross correlation loop.
pivOpts.saveRaw = false;

% Option to use either straightforward gaussian smoothing,
% biorthogonal-spline based wavelet denoising or divergence reduction and fill
% using relax34 algorithm (detailed in my thesis) for intermediate steps of a
% multipass algorithm. Options are 'smooth3', 'p4' and 'relax34'
pivOpts.interPassFilter = 'smooth3';

% ET - specific options
pivOpts.ETcorrSize = [64 32 32];
pivOpts.ETstdDevVoxels = [3 3 3];
pivOpts.ETdistanceCriterion = 0.1;

% Parse variable arguments (containing non-default values) into opts structure
pivOpts = parse_pv_pairs(pivOpts, varargin);
