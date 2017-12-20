function [voxels] = mart_large(setup, cameras, pixels, weighting, num_mart_iters, mu, init_intensity, pix_thresh, vox_thresh)
%MART_LARGE Performs tomographic reconstruction using the MART algorithm
%	This is part of a measurement technique for fluid dynamics research; 
%	Tomographic Particle Image Velocimetry [ref 1]. 
%	Particle fields are captured with multiple cameras, reconstructed in 
%	3D using this routine, then cross-correlated to measure a fully 3D 
%	velocity field within a fluid region.
%
%   This function is a wrapper for the mxmart_large function (written in
%   FORTRAN) which performs the actual reconstruction.
%
%   This is a straightforward implementation of the MART technique described in 
%   ref [1]. Some assumptions are made to reduce computational loading. 
%   These are as follows:
%
%	The MEX file mxmart_large does not include rigorous internal checks. For 
%   safe operation, use this function as a wrapper for it.
%
% Syntax:  
%       [voxels] = mart_large(  setup, cameras, pixels, weighting, ...
%                               num_mart_iters, mu, init_intensity, ...
%                               pix_thresh, vox_thresh)
%
%		Solves the classical Ax = b matrix problem...
%					wij    --> A
%					voxels --> x
%					pixels --> b
%		 ...using the Multiplicative Algebraic Reconstruction Technique.
%
% Inputs:
%
%       setup      	struct      Case setup structure for a given calibration
%                               (see setup_case.m) which contains the following
%                               fields: 'c','d','vox_X','vox_Y','vox_Z'.
%
%		pixels		[npix x 1]  pixels(i) > 0 for all i
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
%		num_mart_iters  [1 x 1]	integer value, 0 < numiters
%								Number of iterations of the MART 
%								algorithm to perform (recommended 5). 
%								The first iteration will take 
%								considerably longer than subsequent ones
%								as zero-level voxels are assigned.
%
%		init_intensity  [1 x 1]	0 <= init_intensity
%								Sets an initial guess for the intensity 
%								of the voxels. Suggest: 
%								mean_pixel_intensity * num_pix / num_vox
%
%		mu              [1 x 1]	0 < mu < 1
%								relaxation constant of the mart 
%								iteration (see ref [1]).
%
%       pix_thresh      [1 x 1] 0 <= pix_thresh < 1
%                               Threshold of intensity below which pixels are
%                               designated to have zero intensity. Usually set
%                               to 0 and allow image preprocessing to zero
%                               pixels which do not contain particle data.
%
%       vox_thresh      [1 x 1] 0 <= vox_thresh < 1
%                               Threshold of intensity below which voxels are
%                               designated to have zero intensity, and are thus
%                               excluded from the computation. Usually set to
%                               2*eps('single').
%
% Outputs:
%		
%		voxels		[nvox x 1]	single
%								Reconstructed intensity of each voxel.
% 
% Other files required:   	mxmart_large.mex*** (MATLAB MEX file)
% Subfunctions:             weightfunction
% MAT-files required:       none
%
% Future Improvements:
%       None foreseen, except expansion to account for alterations made to
%       mxmart_large (see future improvements for that function).
%
% Author:           T.H. Clark
% Work address:     Fluids Lab
%                   Cambridge University Engineering Department
%                   2 Trumpington Street
%                   Cambridge
%                   CB21PZ
% Email:            t.clark@cantab.net
% Website:          http://www.eng.cam.ac.uk/thc29
%
% Created:          10 June 2009 
% Last revised:     02 July 2009

%   Copyright (c) 2007-2015  Thomas H. Clark

%% GET VARIABLES FROM SETUP STRUCTURE

nvox_X = numel(setup.vox_X);
nvox_Y = numel(setup.vox_Y);
nvox_Z = numel(setup.vox_Z);
c = [];
d = [];
los_factor = [];
for i = 1:1:numel(cameras)
    c = [c; setup.c{cameras(i)}]; %#ok<*AGROW>
    d = [d; setup.d{cameras(i)}];
    los_factor = [los_factor, setup.los_factor{cameras(i)}];
end


%% GET WEIGHTING FUNCTION

lookup_wts = weightfunction(weighting);


%% PERFORM RIGOROUS CHECKS

% The following checks are made:
% 1. is lookup_wts the right size (226x1)?
% 2. has the pixels array been masked to the right size?
% 3. are c and d the same size as the pixels array?
% 4. Is the initial intensity > 0?
% 5. Is the number of mart iterations a whole number > 0 ?
% 7. Is lookup_wts nonnegative (>= 0)
% 8. Are the pixel intensities >= 0?
% 9. Are init_intensity, numiters, nvox_X, nvox_Y, nvox_Z, mu and shortcut_flag
%       all scalars?

% Test variable sizes:
if ~isequal(size(init_intensity),[1 1])
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_large.m: init_intensity must be a scalar variable.')
elseif ~isequal(size(num_mart_iters),[1 1])
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_large.m: num_mart_iters must be a scalar variable.')
elseif ~isequal(size(nvox_X),[1 1])
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_large.m: nvox_X must be a scalar variable.')
elseif ~isequal(size(nvox_Y),[1 1])
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_large.m: nvox_Y must be a scalar variable.')
elseif ~isequal(size(nvox_Z),[1 1])
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_large.m: nvox_Z must be a scalar variable.')
elseif ~isequal(size(mu),[1 1])
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_large.m: mu must be a scalar variable.')
elseif ~isequal(size(pix_thresh),[1 1])
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_large.m: pix_thresh must be a scalar variable.')
elseif ~isequal(size(vox_thresh),[1 1])
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_large.m: vox_thresh must be a scalar variable.')
elseif ~isequal(size(lookup_wts),[227 1])
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_large.m: lookup_wts must have size [226 x 1].')
elseif (~isequal(size(pixels,1),size(c,1))) || (~isequal(size(pixels,2),1))
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_large.m: pixels array must have size [P x 1] where c coeffs array is of size [P x 3]. Check that pixels has been masked properly using pixels = pixels(setup_struct.mask).')
elseif ~isequal(size(d),size(c))
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_large.m: Setup structure invalid (size(c) not equal to size(d)).')
elseif ~isequal(size(d,2),3)
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_large.m: Coefficients arrays c and d must have second dimension equal to 3 (size(c,2) = 3).')
elseif ~isequal(size(los_factor,1),2)
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_large.m: los_factors array must have size(los_factors,1) = 2.')
elseif ~isequal(size(los_factor,2),size(c,1))
    error('MATLAB:TomoPIVToolbox:IncorrectSize','mart_large.m: los_factors array must have size(los_factors,2) = P where P is the nnumber of pixels in the reconstruction.')
end

% Test for nonnegative variables
if init_intensity < 0
    error('MATLAB:TomoPIVToolbox:InvalidValue','mxmart_large.m: init_intensity value is negative. Must be positive, recommend init_intensity = 1')
elseif init_intensity == 0
    error('MATLAB:TomoPIVToolbox:InvalidValue','mxmart_large.m: init_intensity value is zero. MART iteration invalid.')
elseif rem(num_mart_iters,round(num_mart_iters)) > eps('double')
    error('MATLAB:TomoPIVToolbox:InvalidValue','mxmart_large.m: number of mart iterations must be an integer value.')
elseif num_mart_iters <= 0
    error('MATLAB:TomoPIVToolbox:InvalidValue','mxmart_large.m: number of mart iterations <= 0. No iteration will occur.')
elseif sum(pixels < 0) > 0
    error('MATLAB:TomoPIVToolbox:InvalidValue','mxmart_large.m: pixel intensity array contains negative values.')
elseif nvox_X <= 0 
    error('MATLAB:TomoPIVToolbox:InvalidValue','mxmart_large.m: nvox_X <= 0. Must be >= 1 for at least one voxel to be reconstructed.')
elseif nvox_Y <= 0 
    error('MATLAB:TomoPIVToolbox:InvalidValue','mxmart_large.m: nvox_Y <= 0. Must be >= 1 for at least one voxel to be reconstructed.')
elseif nvox_Z <= 0 
    error('MATLAB:TomoPIVToolbox:InvalidValue','mxmart_large.m: nvox_Z <= 0. Must be >= 1 for at least one voxel to be reconstructed.')
elseif rem(nvox_X,round(nvox_X)) > eps('double')
    error('MATLAB:TomoPIVToolbox:InvalidValue','mxmart_large.m: nvox_X must be an integer value.')
elseif rem(nvox_Y,round(nvox_Y)) > eps('double')
    error('MATLAB:TomoPIVToolbox:InvalidValue','mxmart_large.m: nvox_Y must be an integer value.')
elseif rem(nvox_Z,round(nvox_Z)) > eps('double')
    error('MATLAB:TomoPIVToolbox:InvalidValue','mxmart_large.m: nvox_Z must be an integer value.')
elseif sum(lookup_wts < 0) > 0
    error('MATLAB:TomoPIVToolbox:InvalidValue','mxmart_large.m: lookup_wts table contains negative values.')
elseif ~isreal(lookup_wts)
    error('MATLAB:TomoPIVToolbox:InvalidValue','mxmart_large.m: lookup_wts table contains complex numbers.')
elseif mu <= 0 
    error('MATLAB:TomoPIVToolbox:InvalidValue','mxmart_large.m: relaxation factor mu is <= 0. Recommend 0.5 for stable MART operation.')
elseif ~(pix_thresh >= eps('single'))
    error('MATLAB:TomoPIVToolbox:InvalidValue','mxmart_large.m: Pixel Intensity Threshold must be >= eps(''single'')')
elseif ~(vox_thresh >= 0)
    error('MATLAB:TomoPIVToolbox:InvalidValue','mxmart_large.m: Voxel Intensity Threshold must be >= 0 (recommend eps(''single'')')
end
    
% Errors on variable values
if nnz(los_factor > 1.0) > 1
    error('MATLAB:TomoPIVToolbox:InvalidValue','mxmart_large.m: maximum LOS factors cannot exceed 1.0')
elseif nnz(los_factor < 0) > 1
    error('MATLAB:TomoPIVToolbox:InvalidValue','mxmart_large.m: minimum LOS factors cannot be less than 0.0')
elseif (pix_thresh >= 1)
    error('MATLAB:TomoPIVToolbox:InvalidValue','mxmart_large.m: Pixel Intensity Threshold is >= 1 (all pixels zeroed)')
elseif (vox_thresh >= init_intensity)
    error('MATLAB:TomoPIVToolbox:InvalidValue','mxmart_large.m: Voxel Intensity Threshold is >= Initial Intensity (0 voxels reconstructed)')
end

% Warnings on variable values (less important)
if num_mart_iters > 5
    warning('MATLAB:TomoPIVToolbox:InvalidValue','mxmart_large.m: number of mart iterations > 5. Iteration may take unnecessarily long.')
elseif (vox_thresh < eps('single')) && (vox_thresh > 0)
    warning('MATLAB:TomoPIVToolbox:InvalidValue','mxmart_large.m: Voxel Intensity Threshold < eps(''single''), but > 0. Setting to eps(''single'')')
    vox_thresh = eps('single');
end
   

%% TYPECAST TO PREVENT SegVs

initial         = single(init_intensity);
numiters        = int32(num_mart_iters);
pixels          = single(pixels);
nvox_X          = int32(nvox_X);
nvox_Y          = int32(nvox_Y);
nvox_Z          = int32(nvox_Z);
c               = single(c);
d               = single(d);
lookup_wts      = single(lookup_wts);
mu              = single(mu);
los_factor      = single(los_factor);
pix_thresh      = single(pix_thresh);
vox_thresh      = single(vox_thresh);


%% CALL THE MEX FILE

[voxels] = mxmart_large(initial, numiters, pixels, nvox_X, ...
						nvox_Y, nvox_Z, c, d, lookup_wts, ...
						mu, los_factor,...
                        pix_thresh, vox_thresh);


end % End function mart_large











