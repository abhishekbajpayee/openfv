function [ux uy uz snr nmr] = PIV_3d_validatefield(nWindows, xGrid, yGrid, zGrid, ux, uy, uz, snr, iPass, pivOpts)
%PIV_3D_VALIDATEFIELD identifies and removes false vectors in a field
%   The vector validation criteria used are the following:
%
%       - normalised median test
%       - maximum displacement test
%       - maximum turbulent intensity test
%       - minimum correlation plane SNR test
%
%   False vectors are replaced with the second peak in the correlation
%   plane where valid
%   
%   Returned velocity fields may still contain NaN elements. These 
%   represent areas in the reconstructions which are poorly seeded, poorly 
%   illuminated, contain complex regions of flow (low SNR cause by multiple
%   correlation peaks in a volume) reflections (from walls etc).
%
%   It is suggested that missing values are replaced using either:
%       - interpolation methods (low pass, linear, etc)
%       - iterative local mean calculation
%       - iterative replace of primary peak vectors following false vector removal
%       - some flow-dependent criterion (e.g POCS constraint that divergence = 0).
%
% Syntax:
%       [ux uy uz snr] = PIV_3d_validatefield(nWindows, ux, uy, uz, snr, iiPass, piv
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
% Author:               T.H. Clark
% Work address:         Fluids Lab
%                       Cambridge University Engineering Department
%                       2 Trumpington Street
%                       Cambridge
%                       CB21PZ
% Email:                t.clark@cantab.net
% Website:              http://cambridge.academia.edu/ThomasClark/
%
% Revision History:     11 July 2011        Based on PIV_3d_vectorcheck,
%                                           modified to work with the fMex
%                                           functions and an SNR input. 
%                       16 July 2011        Altered to remove vectors based
%                                           on one criterion at a time 
%                                           instead of applying all
%                                           criteria in parallel. Hopefully
%                                           this will prevent the nmr test
%                                           from getting muddled by groups
%                                           of large magnitude vectors
%

%   Copyright (c) 2007-2015  Thomas H. Clark

% Get limits etc out of the PIV options structure
max_ux = pivOpts.maxUx;
max_uy = pivOpts.maxUy;
max_uz = pivOpts.maxUz;
eps_PIV_noise = pivOpts.eps_PIV_noise(iPass);
eps_threshold = pivOpts.eps_threshold(iPass);
max_snr= pivOpts.maxSNR;


% Matrix containing allowable turbulent intensities. Apply constraints to the
% cross-stream, streamwise and normal components, based on U only.
turbIntensity = pivOpts.turbIntensity;

% Some peak location algorithms give 'secondary' peaks (i.e. return the peak
% location using some different method, or return locations of multiple peaks in
% the correlation plane). These can be used to fill in false data if present,
% for up to three potential peaks.
nPeaks = size(ux,2);

% Re-sort to straightforward 3D arrays for each fit:

%   Retrieve ux,uy,uz,snr grids for primary peak
ux_pri  = reshape(ux(:,1),  size(xGrid));
uy_pri  = reshape(uy(:,1),  size(xGrid));
uz_pri  = reshape(uz(:,1),  size(xGrid));
snr_pri = reshape(snr(:,1), size(xGrid));

%   Retrieve ux,uy,uz,snr grids for secondary peak
if nPeaks > 1
    ux_sec  = reshape(ux(:,2),  size(xGrid));
    uy_sec  = reshape(uy(:,2),  size(xGrid));
    uz_sec  = reshape(uz(:,2),  size(xGrid));
    snr_sec = reshape(snr(:,2), size(xGrid));
end

% Initialise outputs
ux = ux_pri;
uy = uy_pri;
uz = uz_pri;
snr = snr_pri;
    
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

% Find spurious vectors based on maximum known displacement
spur_mask_disp = PIV_3d_maxdisplacement(ux_pri, uy_pri, uz_pri, max_ux, max_uy, max_uz);
ux(spur_mask_disp) = NaN;
uy(spur_mask_disp) = NaN;
uz(spur_mask_disp) = NaN;

% Find spurious vectors based on known turbulent intensities
spur_mask_turb = PIV_3d_turbulentintensity(ux_pri, uy_pri, uz_pri, turbIntensity, velocityUVW);
ux(spur_mask_turb) = NaN;
uy(spur_mask_turb) = NaN;
uz(spur_mask_turb) = NaN;

% Find spurious vectors based on primary peak velocities
[spur_mask_pk nmr] = PIV_3d_normalisedmedian(ux_pri, uy_pri, uz_pri, eps_PIV_noise, eps_threshold);
ux(spur_mask_pk) = NaN;
uy(spur_mask_pk) = NaN;
uz(spur_mask_pk) = NaN;

% Find spurious vectors based on maximum snr
spur_mask_snr = PIV_3d_maxsnr(snr_pri, max_snr);

% Logical combination of the criteria:
spur_mask = spur_mask_pk | spur_mask_disp | spur_mask_turb | spur_mask_snr;


% Replace spurious vectors using secondary correlation peaks
% if nPeaks > 1
%     ux(spur_mask)  = ux_sec(spur_mask);
%     uy(spur_mask)  = uy_sec(spur_mask);
%     uz(spur_mask)  = uz_sec(spur_mask);
%     snr(spur_mask) = snr_sec(spur_mask);
%     
%     % Repeat spurious vector calculation combined velocity distribution.
%     % Test outcomes applied in parallel rather than sequential
%     spur_mask_disp = PIV_3d_maxdisplacement(   ux, uy, uz, max_ux, max_uy, max_uz);
%     spur_mask_turb = PIV_3d_turbulentintensity(ux, uy, uz, turbIntensity, velocityUVW);
%     [spur_mask_pk nmr]   = PIV_3d_normalisedmedian(  ux, uy, uz, eps_PIV_noise, eps_threshold);
%     spur_mask_snr  = PIV_3d_maxsnr(snr, max_snr);
%     spur_mask      = spur_mask_pk | spur_mask_disp | spur_mask_turb | spur_mask_snr;
%     
% end

% INSERT ITERATIVE REPLACEMENT ROUTINE HERE

% Any velocities still marked as spurious should be set to NaN.
ux(spur_mask) = NaN;
uy(spur_mask) = NaN;
uz(spur_mask) = NaN;

















