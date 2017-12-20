function [voxels] = mart_mfg(setup, cameras, pixels, weighting, nIterations, mu, pix_thresh)
%MART_MFG Performs tomographic reconstruction using MFG-MART algorithm
%	
%   This function is a wrapper for the mxmart_mfg mex function (written in
%   FORTRAN) which performs the actual reconstruction.
%
%   This is an implementation of the Multiplicative First Guess technique
%   [Ref 1] with an option for further Multiplicative Algebraic Reconstruction 
%   Technique (MART) iterations [Ref 2] following the MFG.
%
%	The MEX file mxmart_mfg does not include rigorous internal checks. For 
%   safe operation, use this function as a wrapper for it.
%
% Syntax:  
%       [voxels] = mart_mfg(setup, cameras, pixels, weighting, nIterations, mu)
%
% Inputs:
%
%       setup       struct      Case setup structure for a given calibration
%                               (see setup_case.m) which contains the following
%                               fields: 'c','d','vox_X','vox_Y','vox_Z'.
%
%       cameras     [1 x nCams] Number array of cameras to use in the
%                               reconstruction, typically [1 2 3 4].
%
%		pixels		[npix x 1]	pixels(i) > 0 for all i
%								Contains the intensities of the pixels 
%								from which the reconstruction is made. NB this
%								must already be masked (by a mask also found in
%								the setup struct) to limit this vector to the
%								pixels relevant to this setup.
%
%       weighting 	string      'circle','gaussian'
%                               Suggest 'circle'. This is the function that
%                               is applied to determine the weighting between a
%                               pixel-voxel combination, given the distance
%                               between them. See figure 'weighting_fcn.fig' for
%                               display.
%
%		nIterations  [1 x 1]	integer value, 0 < numiters
%								Number of iterations of the MART 
%								algorithm to perform (recommended 5). 
%								The first iteration will take 
%								considerably longer than subsequent ones
%								as zero-level voxels are assigned.
%
%		mu         	[1 x 1]		0 < mu < 1
%								relaxation constant of the mart 
%								iteration (see ref [1]).
%   
%       pix_thresh  [1 x 1]     0 <= pix_thresh
%                               Allows acceleration of the solution by 
%                               effectively zeroing and ignoring all 
%                               pixels with intensity below pix_thresh
%
%
% Outputs:
%		
%		voxels		[nvox x 1]	single
%								Reconstructed intensity of each voxel.
% 
% Other files required:   	mxmart_mfg.mex*** (MATLAB MEX file)
% Subfunctions:             weightfunction
% MAT-files required:       none
%
% Future Improvements:
%       None foreseen, except expansion to account for alterations made to
%       mxmart_large (see future improvements for that function).
%
% Author:               T.H. Clark
% Work address:         Fluids Lab
%                       Cambridge University Engineering Department
%                       2 Trumpington Street
%                       Cambridge
%                       CB21PZ
% Email:                t.clark@cantab.net
% Website:              http://www.eng.cam.ac.uk/thc29
%
% Revision History:     22 June 2009        Created
%                       30 July 2011        Got rid of initial intensity
%                                           and shortcut flag variables
%                                           (unused). Altered errors to
%                                           have proper error code.
%                                           Commented out old debugging
%                                           code.

%   Copyright (c) 2007-2015  Thomas H. Clark

%% GET VARIABLES FROM SETUP STRUCTURE

nvox_X      = numel(setup.vox_X);
nvox_Y      = numel(setup.vox_Y);
nvox_Z      = numel(setup.vox_Z);
c           = [];
d           = [];
los_factor  = [];
cam_offsets = zeros(numel(cameras),1);
for i = 1:1:numel(cameras)
    c               = [c; setup.c{cameras(i)}]; %#ok<AGROW>
    d               = [d; setup.d{cameras(i)}]; %#ok<AGROW>
    cam_offsets(i)  = size(c,1);
    los_factor      = [los_factor, setup.los_factor{cameras(i)}]; %#ok<AGROW>
end


%% GET WEIGHTING FUNCTION

lookup_wts = weightfunction(weighting);


%% PERFORM RIGOROUS CHECKS

% The following checks are made:
% 1. Is lookup_wts the right size (226x1)?
% 2. Has the pixels array been masked to the right size?
% 3. Are c and d the same size as the pixels array?
% 4. Is the number of mart iterations a whole number > 0 ?
% 5. Is lookup_wts nonnegative (>= 0)
% 6. Are the pixel intensities >= 0?
% 7. Are numiters, nvox_X, nvox_Y, nvox_Z, and mu all scalars?

% Test variable sizes:
if ~isequal(size(nIterations),[1 1])
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_mfg.m: nIterations must be a scalar variable.')
elseif ~isequal(size(pix_thresh),[1 1])
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_mfg.m: pix_thresh must be a scalar variable.')
elseif ~isequal(size(nvox_X),[1 1])
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_mfg.m: nvox_X must be a scalar variable.')
elseif ~isequal(size(nvox_Y),[1 1])
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_mfg.m: nvox_Y must be a scalar variable.')
elseif ~isequal(size(nvox_Z),[1 1])
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_mfg.m: nvox_Z must be a scalar variable.')
elseif ~isequal(size(mu),[1 1])
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_mfg.m: mu must be a scalar variable.')
elseif ~isequal(size(lookup_wts),[227 1])
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_mfg.m: lookup_wts must have size [226 x 1].')
elseif (~isequal(size(pixels,1),size(c,1))) || (~isequal(size(pixels,2),1))
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_mfg.m: pixels array must have size [P x 1] where c coeffs array is of size [P x 3]. Check that pixels has been masked properly using pixels = pixels(setup_struct.mask).')
elseif ~isequal(size(d),size(c))
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_mfg.m: Setup structure invalid (size(c) not equal to size(d)).')
elseif ~isequal(size(d,2),3)
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_mfg.m: Coefficients arrays c and d must have second dimension equal to 3 (size(c,2) = 3).')
elseif ~isequal(size(los_factor,1),2)
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_mfg.m: los_factors array must have size(los_factors,1) = 2.')
elseif ~isequal(size(los_factor,2),size(c,1))
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_mfg.m: los_factors array must have size(los_factors,2) = P where P is the nnumber of pixels in the reconstruction.')
end

% Test for nonnegative variables
if rem(nIterations,round(nIterations)) > eps('double')
    error('MATLAB:TomoPIVToolbox:InvalidValue','mart_mfg.m: number of mart iterations must be an integer value.')
%elseif nIterations <= 0
%    error('MATLAB:TomoPIVToolbox:','mart_mfg.m: number of mart iterations <= 0. No iteration will occur.')
elseif sum(pixels < 0) > 0
    error('MATLAB:TomoPIVToolbox:InvalidValue','mart_mfg.m: pixel intensity array contains negative values.')
elseif nvox_X <= 0 
    error('MATLAB:TomoPIVToolbox:InvalidValue','mart_mfg.m: nvox_X <= 0. Must be >= 1 for at least one voxel to be reconstructed.')
elseif nvox_Y <= 0 
    error('MATLAB:TomoPIVToolbox:InvalidValue','mart_mfg.m: nvox_Y <= 0. Must be >= 1 for at least one voxel to be reconstructed.')
elseif nvox_Z <= 0 
    error('MATLAB:TomoPIVToolbox:InvalidValue','mart_mfg.m: nvox_Z <= 0. Must be >= 1 for at least one voxel to be reconstructed.')
elseif rem(nvox_X,round(nvox_X)) > eps('double')
    error('MATLAB:TomoPIVToolbox:InvalidValue','mart_mfg.m: nvox_X must be an integer value.')
elseif rem(nvox_Y,round(nvox_Y)) > eps('double')
    error('MATLAB:TomoPIVToolbox:InvalidValue','mart_mfg.m: nvox_Y must be an integer value.')
elseif rem(nvox_Z,round(nvox_Z)) > eps('double')
    error('MATLAB:TomoPIVToolbox:InvalidValue','mart_mfg.m: nvox_Z must be an integer value.')
elseif sum(lookup_wts < 0) > 0
    error('MATLAB:TomoPIVToolbox:InvalidValue','mart_mfg.m: lookup_wts table contains negative values.')
elseif mu <= 0 
    error('MATLAB:TomoPIVToolbox:InvalidValue','mart_mfg.m: relaxation factor mu is <= 0. Recommend 0.5 for stable MART operation.')
elseif pix_thresh < 0 
    error('MATLAB:TomoPIVToolbox:InvalidValue','mart_mfg.m: pixel zero threshold pix_thresh must be >= 0.')
end
    
% Errors on variable values
if nnz(los_factor > 1.0) > 1
    error('MATLAB:TomoPIVToolbox:InvalidValue','mart_mfg.m: maximum LOS factors cannot exceed 1.0')
elseif nnz(los_factor < 0) > 1
    error('MATLAB:TomoPIVToolbox:InvalidValue','mart_mfg.m: minimum LOS factors cannot be less than 0.0')
end


% Warnings on variable values (less important)
if nIterations > 5
    warning('MATLAB:TomoPIVToolbox:Value','mart_mfg.m: number of mart iterations > 5. Iteration may take unnecessarily long.')
end
   

%% TYPECAST TO PREVENT SegVs

numiters        = int32(nIterations);
pixels          = single(pixels);
nvox_X          = int32(nvox_X);
nvox_Y          = int32(nvox_Y);
nvox_Z          = int32(nvox_Z);
c               = single(c);
d               = single(d);
lookup_wts      = single(lookup_wts);
mu              = single(mu);
los_factor      = single(los_factor);
cam_offsets     = int32(cam_offsets);
pix_thresh      = single(pix_thresh);


% Debug: save inputs
save debugMFG.mat

%% CALL THE MEX FILE
tic
[voxels] = mxmart_mfg( numiters, pixels, nvox_X, ...
                       nvox_Y, nvox_Z, c, d, lookup_wts, ...
                       mu, los_factor, cam_offsets, pix_thresh);
disp(['    mart_mfg: Reconstruction made in ' num2str(toc) 's'])

% TEST CODE
% toc
% save result_mfg setup voxels
%                         
% 
% %% TYPECAST TO PREVENT SegVs
% 
% initial         = single(init_intensity);
% numiters        = int32(nIterations);
% pixels          = single(pixels);
% nvox_X          = int32(nvox_X);
% nvox_Y          = int32(nvox_Y);
% nvox_Z          = int32(nvox_Z);
% c               = single(c);
% d               = single(d);
% lookup_wts      = single(lookup_wts);
% mu              = single(mu);
% shortcut_flag   = int32(0);
% los_factor      = single(los_factor);
% 
% 
% %% CALL THE MEX FILE
% tic
% [voxels] = mxmart_large(initial, numiters, pixels, nvox_X, ...
% 						nvox_Y, nvox_Z, c, d, lookup_wts, ...
% 						mu, shortcut_flag, los_factor);
%                     toc
% save result_large setup voxels



end % End function mart_mfg











