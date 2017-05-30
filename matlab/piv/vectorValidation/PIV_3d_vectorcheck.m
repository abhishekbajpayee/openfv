function [ux uy uz] = PIV_3d_vectorcheck(n_windows, pk_locs_x, pk_locs_y, pk_locs_z, pass, pivOpts)
%PIV_3D_VECTORCHECK locates and iteratively replaces false vectors in a field
%   The vector validation criterion used is the normalised median test
%   (a widely applicable and robust validation test).
%
%   Returned velocity fields may still contain NaN elements. These represent
%   gaps in the flow which arise usually due to poorly seeded areas, or
%   reflections (from walls etc).
%
%   It is suggested that missing values are replaced using either:
%       - interpolation methods (low pass, linear, etc)
%       - iterative local mean calculation (similar to linear interpolation)
%       - some flow-dependent criterion (e.g constraint that divergence = 0).
%
% Syntax:
%       [ux uy uz] = PIV_3d_vectorcheck(n_windows, pk_locs_x, pk_locs_y, pk_locs_z, pivOpts)
%       
% Inputs:
%
% Outputs:
%
% Examples:
%
%   See PIV_3d.m
%
% Future Improvements:       
% Other m-files required:   none
% Subfunctions:             none
% Nested functions:         none
% MAT-files required:       none
%
%
% Author:           T.H. Clark
% Work address:     Fluids Lab
%                   Cambridge University Engineering Department
%                   2 Trumpington Street
%                   Cambridge
%                   CB21PZ
% Email:            t.clark@cantab.net
% Website:          http://cambridge.academia.edu/ThomasClark/
%
% Created:          19 October 2009
% Last revised:     08 July 2010

%   Copyright (c) 2007-2015  Thomas H. Clark

max_ux = pivOpts.maxUx;
max_uy = pivOpts.maxUy;
max_uz = pivOpts.maxUz;
eps_PIV_noise = pivOpts.eps_PIV_noise(pass);
eps_threshold = pivOpts.eps_threshold(pass);

% Matrix containing allowable turbulent intensities. Apply constraints to the
% cross-stream, streamwise and normal components, based on U only.
turbIntensity = pivOpts.turbIntensity;

% Storage of peak locations:
%
%   Peak locations (velocities) are stored in 4-D arrays, of dimension 
%       [3 x n_windows(2) x n_windows(1) x n_windows(3)]
%   The first dimension is 3, as the array contains results from primary,
%   secondary and tertiary peak fitting.

% Some peak location algorithms give 'secondary' peaks (i.e. return the peak
% location using some different method, or return locations of multiple peaks in
% the correlation plane). These can be used to fill in false data if present,
% for up to three potential peaks.
nPeaks = size(pk_locs_x,1);

% Re-sort to straightforward 3D arrays for each fit:

%   Retrieve x,y,z components for primary peak
ux_pri = reshape(pk_locs_x(1,:,:,:), n_windows);
uy_pri = reshape(pk_locs_y(1,:,:,:), n_windows);
uz_pri = reshape(pk_locs_z(1,:,:,:), n_windows);

%   Retrieve x,y,z components for secondary peak
if nPeaks > 1
    ux_sec = reshape(pk_locs_x(2,:,:,:), n_windows);
    uy_sec = reshape(pk_locs_y(2,:,:,:), n_windows);
    uz_sec = reshape(pk_locs_z(2,:,:,:), n_windows);
end

%   Retrieve x,y,z components for tertiary peak
if nPeaks > 2
    ux_ter = reshape(pk_locs_x(3,:,:,:), n_windows);
    uy_ter = reshape(pk_locs_y(3,:,:,:), n_windows);
    uz_ter = reshape(pk_locs_z(3,:,:,:), n_windows);
end

%
% IF UNCOMMENTING FOR DEBUGGING, YOU'LL NEED TO PASS IN THE mesh_x ETC VARIABLES
% fh = figure(214);
% clf
% set(fh,'NumberTitle', 'off');
% set(fh,'Name','PIV_3d_vectorcheck Debugging');
% 
% quiver3(mesh_x, mesh_y, mesh_z, ux_pri, uy_pri, uz_pri,'b')
% hold on
% quiver3(mesh_x, mesh_y, mesh_z, ux_sec, uy_sec, uz_sec,'r')
% quiver3(mesh_x, mesh_y, mesh_z, ux_ter, uy_ter, uz_ter,'g')
% drawnow

% Determine mean velocities if necessary
velocityUVW = pivOpts.meanVelocity;
if isnan(velocityUVW(1))
	velocityUVW(1) = nanmedian(ux_pri(:));
end
if isnan(velocityUVW(2))
	velocityUVW(2) = nanmedian(uy_pri(:));
end
if isnan(velocityUVW(3))
	velocityUVW(3) = nanmedian(uz_pri(:));
end

% Find spurious vectors based on known turbulent intensities
spur_mask_turb = PIV_3d_turbulentintensity(ux_pri, uy_pri, uz_pri, turbIntensity, velocityUVW);

% Find spurious vectors based on primary peak velocities
spur_mask_pk = PIV_3d_normalisedmedian(ux_pri, uy_pri, uz_pri, eps_PIV_noise, eps_threshold);

% Find spurious vectors based on maximum known displacement
spur_mask_disp = PIV_3d_maxdisplacement(ux_pri, uy_pri, uz_pri, max_ux, max_uy, max_uz);

% Logical combination of the two criteria:
spur_mask = spur_mask_pk | spur_mask_disp | spur_mask_turb;
% spur_mask = spur_mask_turb;


% Initialise output velocity distributions
ux = ux_pri;
uy = uy_pri;
uz = uz_pri;


% Replace spurious vectors using secondary correlation peaks
if nPeaks > 1
    ux(spur_mask) = ux_sec(spur_mask);
    uy(spur_mask) = uy_sec(spur_mask);
    uz(spur_mask) = uz_sec(spur_mask);
    
    % Repeat spurious vector calculation combined velocity distribution
    spur_mask_turb = PIV_3d_turbulentintensity(ux, uy, uz, turbIntensity, velocityUVW);
    spur_mask_pk = PIV_3d_normalisedmedian(ux, uy, uz, eps_PIV_noise, eps_threshold);
    spur_mask_disp = PIV_3d_maxdisplacement(ux, uy, uz, max_ux, max_uy, max_uz);
    spur_mask = spur_mask_pk | spur_mask_disp | spur_mask_turb;
end


% Replace using tertiary correlation peaks
if nPeaks > 2
    ux(spur_mask) = ux_ter(spur_mask);
    uy(spur_mask) = uy_ter(spur_mask);
    uz(spur_mask) = uz_ter(spur_mask);
    
    % Any remaining spurious vectors can be filled in if necessary using a variety
    % of interpolation techniques. However, that is not done in this routine - it
    % may be done later in order to fill in 'gaps' in the data, or not (in order to
    % highlight regions of flow where the PIV is not working well). However, in
    % order to do it we need to get rid of the spurious vectors:
    spur_mask_turb = PIV_3d_turbulentintensity(ux, uy, uz, turbIntensity, velocityUVW);
    spur_mask_pk = PIV_3d_normalisedmedian(ux, uy, uz, eps_PIV_noise, eps_threshold);
    spur_mask_disp = PIV_3d_maxdisplacement(ux, uy, uz, max_ux, max_uy, max_uz);
    spur_mask = spur_mask_pk | spur_mask_disp | spur_mask_turb;
end

% Any velocities still marked as spurious should be set to NaN.
ux(spur_mask) = NaN;
uy(spur_mask) = NaN;
uz(spur_mask) = NaN;

















