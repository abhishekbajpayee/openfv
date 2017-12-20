function [voxelsOutput tRecon varargout] = tomo(setup, cameras, processedImgs, tomoOpts)
%TOMO Harness/interface function for tomographic reconstruction algorithms
%
% Syntax:  
%       [voxels] = tomo(setup, cameras, processedImgs, tomoOpts)
%               Performs tomographic reconstruction using the algorithm and
%               parameter options defined in tomoOpts.
%       [voxels tRecon] = tomo(...)
%               Outputs time to perform reconstruction
%       [voxels tRecon residual] = tomo(...)
%               Outputs residual of the wrs solution (utilising wrs
%               reconstruction only, otherwise error).
%
% Inputs:
%       
%       setup         structure     TomoPIV Toolbox standard setup
%                                   structure
%
%       cameras       [1 x nCams]   Camera numbers to reconstruct from
%
%	processedImgs {nCams x nRecons}   cell
%                                   Contains preprocessed images for the
%                                   camera numbers in the 'cameras'
%                                   variable. nRecons is the number of
%                                   reconstructions to make
%
%       tomoOpts        structure   TomoPIV Toolbox standard tomo options
%                                   structure (see help defineTomoOptions)
%
% Outputs:
%       
%       voxels        [nvox_Y x nvox_X x nvox_Z] single
%                                   Voxels array(s) ready for visualisation or
%                                   cross correlation in tomoPIV standard
%                                   voxels array form; reshaped to 3D
%                                   array.
%                               or
%                     {nRecons x 1} cell
%                                   Cell containing voxels arrays as above
%                                   for each of the n reconstructions
%
%       tRecon        [1 x 1]       Time in seconds to make the
%                                   reconstruction
%
%       residual      [1 x 1] or nIters x 1]
%                                   For wrs methods 'simplematlab' and
%                                   'preconditionedmatlab', the residual
%                                   vector (for each iteration) is output.
%                                   For 'lsqlin' and 'lsqnonneg' methods,
%                                   the final residual value is output.
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
% Revision History:        	01 August 2011      Created
%                           13 August 2011      Modified for batch
%                                               reconstructions.

%   Copyright (c) 2007-2015  Thomas H. Clark

% Start timing counter
tic


%% MASK PIXELS FROM PROCESSED IMAGES


% Mask the input images to the pixels we require and append to pixels
% vector
pixels = [];
for iCam = cameras
    pixels = [pixels; processedImgs{iCam,1}(setup.mask{iCam})]; %#ok<AGROW>
end

% Extend the pixels array out where multiple reconstructions are required
if (size(processedImgs,2) > 1) && (~strcmpi(tomoOpts.algorithm,'mart_parbatch'))
    error('MATLAB:TomoPIVToolbox:InvalidInput','tomo.m: The processedImgs array has more than one column, but a parallelised reconstruction routine is not specified. please resolve this (either reconstruct one at a time, or specify use of mart_parbatch)')
end
if size(processedImgs,2) > 1
    for iFrame = 2:size(processedImgs,2)
        pixtemp = [];
        for iCam = cameras
            pixtemp = [pixtemp; processedImgs{iCam,iFrame}(setup.mask{iCam})]; %#ok<AGROW>
        end
        pixels(:,iFrame) = pixtemp; %#ok<AGROW>
    end
end

        


%% PERFORM TOMOGRAPHIC RECONSTRUCTION

switch lower(tomoOpts.algorithm)
    
    case 'mart'
        
        % Use MART algorithm to perform reconstruction
        [voxels] = mart_large(setup, cameras, pixels, ...
                                    tomoOpts.weighting, ...
                                    tomoOpts.nMartIters, ...
                                    tomoOpts.muMart, ...
                                    1, ...
                                    tomoOpts.pixelThreshMart, ...
                                    tomoOpts.voxelThreshMart);
        
    case 'mart_pvr'
        
        % Use MART algorithm with pvr correction to perform reconstruction
        [voxels] = mart_large_pvr(setup, cameras, pixels, ...
                                    tomoOpts.weighting, ...
                                    tomoOpts.nMartIters, ...
                                    tomoOpts.muMart, ...
                                    1, ...
                                    tomoOpts.pixelThreshMart, ...
                                    tomoOpts.voxelThreshMart);
                                
    case 'mart_parbatch'
        
        % Use MART algorithm with pvr correction to perform reconstruction
        [voxels] = mart_parbatch(setup, cameras, pixels, ...
                                    tomoOpts.weighting, ...
                                    tomoOpts.nMartIters, ...
                                    tomoOpts.muMart, ...
                                    1, ...
                                    tomoOpts.pixelThreshMart, ...
                                    tomoOpts.voxelThreshMart);
        
    case 'mfg_mart'

        % Use MART algorithm to perform reconstruction
        [voxels] = mart_mfg(setup, cameras, pixels, ...
                                    tomoOpts.weighting, ...
                                    tomoOpts.nMfgIters, ...
                                    tomoOpts.muMfg, ...
                                    tomoOpts.pixelThreshMfg);

    case 'wrs'
        [voxels residual] = tomoWrs(setup, cameras, pixels, tomoOpts);
        if nargout > 2
            varargout{1} = residual;
        end
        
    otherwise
        error('MATLAB:TomoPIVToolbox:InvalidInput','tomo.m: unrecognised ''algorithm'' string in tomo options structure. Try: ''mart'',''mart_pvr'',''mart_parbatch'',''mfg_mart'' or ''wrs''.')
        
end


%% SAVE VOXELS ARRAY IF REQUIRED

if ~strcmpi(tomoOpts.algorithm,'mart_parbatch')
    
    if isnumeric(tomoOpts.storeResult) && (tomoOpts.storeResult == 1)

        % Then use file selection GparametersUI to save
        tomoPIV_savevoxels(voxels);

    elseif ischar(tomoOpts.storeResult)

        % Save to the filename in tomoOpts.storeResult
        tomoPIV_savevoxels(voxels,tomoOpts.storeResult);

    end
else
    if isnumeric(tomoOpts.storeResult) && (tomoOpts.storeResult == 1)

        % Then use file selection GUI to save
        tomoPIV_savevoxelsbatch(voxels);

    elseif ischar(tomoOpts.storeResult)

        % Save to the filename in tomoOpts.storeResult
        tomoPIV_savevoxelsbatch(voxels,tomoOpts.storeResult);

    end
end


%% RESHAPE AND OUTPUT VOXELS ARRAY

% Reshape to correct size
nvox_X = numel(setup.vox_X);
nvox_Y = numel(setup.vox_Y);
nvox_Z = numel(setup.vox_Z);

% Place into a cell then reshape if batch processed
if strcmpi(tomoOpts.algorithm,'mart_parbatch')
    
    % We use a mex function to do this, as it requires too much memory. The
    % mex function (thanks to James Tursa on the Matlab Newsgroup) strips
    % off the last column of an array without making a dumb copy of the
    % array. Thus we can use more than half the memory in the system.
    voxelsOutput = cell(size(voxels,2),1);
    for iCell = size(voxels,2):-1:1
        voxelsOutput{iCell} = reshape(voxels(:,end), nvox_Y, nvox_X, nvox_Z);
        deletelastcolumn(voxels)
    end
else
    % Otherwise simply reshape
    voxelsOutput = reshape(voxels, nvox_Y, nvox_X, nvox_Z);
end

% Finish timing counter
tRecon = toc;




