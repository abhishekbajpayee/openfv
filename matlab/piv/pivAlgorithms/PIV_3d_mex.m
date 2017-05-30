function [velocityField] = PIV_3d_mex(fieldA, fieldB, varargin)
%PIV_3D_MEX Particle Image Velocimetry using 3D scalar intensity (particle) fields
%   Performs windowed cross correlation of fieldA (particle field at time t)
%   with fieldB (particle field at time t+dt).
%
%   The cross correlation is performed using 3D FFTs and either gaussian or 
%   whittaker peak finding algorithm to determine the correlation displacement.
%   Displacement values are scaled by the time in seconds dt to give
%   output velocities in units voxels/s.
%
%   Window size can be varied. Multi-pass analysis is available, with varying
%   window size and false vector removal options. Successive passes use the
%   previous pass (heavily smoothed) as a first guess, and a window-deformation
%   algorithm is used.
%
% Syntax:
%       [velocityField] = PIV_3d_mex(fieldA, fieldB, dt, pivOpts)
%            Cross-correlates field A (particle field at time t) with field B
%            (particle field at time t + dt)
%
%       [velocityField] = PIV_3d_mex_o2(fieldA, fieldB, fieldC, fieldD, dt, pivOpts)
%            Performs second order accurate cross correlation:
%                  corr = (AxB) + 2(BxC) + (CxD)
%
% Inputs:
%
%       fieldA, fieldB          [nVoxY x nVoxZ x nVoxZ] single
%       fieldC, fieldD                          Contains the reconstructed
%                                               intensity field at a point in
%                                               time, where A,B,C,D represent a
%                                               consecutive time series, each
%                                               frame separated by time dt.
%
%       dt                      [1 x 1] double  The time in seconds between
%                                               successive reconstruction
%                                               instances.
%
%       pivOpts                 structure       Contains the options for
%                                               specifying which PIV algorithm
%                                               to use and the behaviour of the
%                                               algorithm, false vector
%                                               selection parameters, multipass
%                                               options etc. This structure is
%                                               created using the 
%                                                   definePIVOptions() function.
%                                               See help definePIVOptions
%                                               for further details.
%
% Outputs:
%
%       velocityField          structure(nPasses)
%                                               A structure containing the
%                                               intermediate and final velocity
%                                               fields produced by PIV_3d. This
%                                               structure can be used in further
%                                               plotting, analysis and
%                                               postprocessing of the results.
%
% Note on Units:
%
%       Ouptut velocities are in voxels/second. The outputs (in the velocity
%       structure) containing the window centres are in voxel units relative to
%       the local reconstruction array. Use the PIV_3d routine to call this
%       function, as PIV_3d updates the velocity structure to give the outputs
%       in correct units.
%
% References:
%
%   [1] Raffel M. Willert C. Wereley S. and Kompenhans J. Particle Image
%       Velocimetry A Practical Guide (Second Edition). Springer 2007 
%       ISBN 978-3-540-72307-3
%
%   [2] Thomas M. Misra S. Kambhamettu C. and Kirby J.T. (2005)
%       A robust motion estimation algorithm for PIV
%
% Future Improvements:
%
%   [1] More of this code implemented in C. If the peak location and field
%       interpolation could be implemented in C, then this would be advantageous
%       because the entire loop could be set up, preventing the transfer of the
%       intensity fields to the MEX file at each step.
%
%   [2] Alternative to 1, use persistent variables in the MEX file.
%
%   [3] Additions to implement the technique in ref [2]
%       
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
% Revison History:          16 April 2011       Created
%                           17 April 2011       Wrote the code structure and
%                                               carved out parts of
%                                               PIV_3d_matlab which are useful
%                           09 July 2011        Modified for fPIV
%                           12 July 2011        Added second order
%                                               functionality, debugged
%                                               window convection
%                                               calculation and moved the
%                                               calcuation to the
%                                               windowCornerConvect
%                                               routine.

%   Copyright (c) 2007-2015  Thomas H. Clark

%% SORT VARIABLE INPUTS

if nargin == 4
    order   = 'first';
    dt      = varargin{1};
    pivOpts = varargin{2};
else
    order   = 'second';
    fieldC  = varargin{1};
    fieldD  = varargin{2};
    dt      = varargin{3};
    pivOpts = varargin{4};
end


%% PRELIMINARY CALCULATIONS AND SETUP

% Initialise the results structure
emptycell      = cell(pivOpts.nPasses,1);
% velocityField = struct( 'ux',           emptycell, ...
%                         'uy',           emptycell, ...
%                         'uz',           emptycell, ...
%                         'peak_locs',    emptycell, ...
%                         'peak_vals',    emptycell, ...
%                         'peak_void',    emptycell, ...
%                         'win_ctrs_x',   emptycell, ...
%                         'win_ctrs_y',   emptycell, ...
%                         'win_ctrs_z',   emptycell);
if strcmpi(order,'first') 
    velocityField = struct( 'ux',           emptycell, ...
                            'uy',           emptycell, ...
                            'uz',           emptycell, ...
                            'nanMask',      emptycell, ...
                            'snr',          emptycell, ...
                            'winCtrsX',     emptycell, ...
                            'winCtrsY',     emptycell, ...
                            'winCtrsZ',     emptycell, ...
                            'distanceUnits',emptycell,...
                            'medResidual',  emptycell);
else
    velocityField = struct( 'ux',           emptycell, ...
                            'uy',           emptycell, ...
                            'uz',           emptycell, ...
                            'nanMask',      emptycell, ...
                            'snr',          emptycell, ...
                            'winCtrsX',     emptycell, ...
                            'winCtrsY',     emptycell, ...
                            'winCtrsZ',     emptycell, ...
                            'distanceUnits',emptycell,...
                            'medResidual',  emptycell);
end

% Size of voxels arrays
nvox_X = size(fieldA,2);
nvox_Y = size(fieldA,1);
nvox_Z = size(fieldA,3);


% CODE STRUCTURE

% Convert first guess to a velocity field

% Loop for each pass

    % Determine current window centres
        
    % Determine corresponding window corner locations (unshifted)
    
    % Convect window corner locations according to the input velocity field,
    % which must have been converted to the units voxels/second
    
    % Check for out-of-bounds errors and iteratively decrease translation
    % until there are no OOB errors.
    
    % Call fMexPIV
        
    % Perform vector validation
    
    % Store results in TomoPIV Toolbox format
    
    % If not the final pass, smooth the velocity field ready for next pass
    
% end


% Display progress
disp('_______________________________________________________________')
disp(' ')
disp('    PIV_3d:fMexPIV')
disp(' ')
disp('    Running with options:')
disp( pivOpts )
tic



% LOOP FOR EACH PASS
for iPass = 1:pivOpts.nPasses

    % Display progress
    disp(['    Running pass ' num2str(iPass) ' of ' num2str(pivOpts.nPasses) '...'])
    
    
    %% GET PARAMETERS FOR THIS PASS
    
    % Window sizes for current pass...
    wSize = pivOpts.wSize(iPass,:);
    if size(wSize,2) == 1
        wSize = [wSize wSize wSize]; %#ok<AGROW>
    elseif ~isequal(wSize(1),wSize(2),wSize(3))
        error('MATLAB:TomoPIVToolbox:InvalidInput','For MEX and CUDA based PIV algorithms, window size must be the same in all directions (cuboid windows)')
    end
    
    % Overlap
    overlap = pivOpts.overlap(iPass);
    if (overlap < 0) || (overlap >= 100)
        error('MATLAB:TomoPIVToolbox:InvalidInput','Overlap cannot be < 0 or >= 100 %') %#ok<CTPCT>
    end
    
    % Edge cut (number of voxels to shave off all faces of the volume)
    edgeCut = pivOpts.edgeCut;
    
        
    

    %% DETERMINE WINDOW POSITIONS FOR CURRENT PASS
    
    % Max and min locations (in local voxel coords)
    minCoord =                          wSize/2 + 0.5 + edgeCut;
    maxCoord = [nvox_X nvox_Y nvox_Z] - wSize/2 + 0.5 - edgeCut;
    
    % Window spacing in voxels in the X,Y,Z directions
    winSpacing = wSize * (1 - (overlap/100));
    
    % Window Centre Locations
    %   So windows are spaced winSpacing number of voxels apart. The coordinate
    %   of the centre of the first window in each direction is minCoord, and the
    %   maximum coordinate possible is maxCoord.
    wCtrsX = minCoord(1):winSpacing(1):maxCoord(1);
    wCtrsY = minCoord(2):winSpacing(2):maxCoord(2);
    wCtrsZ = minCoord(3):winSpacing(3):maxCoord(3);
    
    % Number of windows in each direction (and in total)
    nWindows = [numel(wCtrsX) numel(wCtrsY) numel(wCtrsZ)];
    nW = prod(nWindows);
    
    % Form 3D meshes of these coordinates
    [wCtrsXGrid wCtrsYGrid wCtrsZGrid] = ndgrid(wCtrsX,wCtrsY,wCtrsZ);
        
    % Determine corresponding window corner locations (unshifted). 
    % There are eight corners to each cube window - Look at the file
    % windowOrdering.jpg in
    %       tomoPIVRoot/code_crosscorrelation/pivAlgorithms/cuda/ 
    % to see the order in which the windows must be arranged for use by the
    % texture fetching algorithm (window deformation).
    % In that diagram, the:
    %   f1 direction increases with increasing column in the voxels array
    %   f2 direction increases with increasing row in the voxels array
    %   f3 direction increases with increasing page in the voxels array
    %
    % Indices into the voxels array are a local frame of reference oriented to
    % the indexing of the voxels array. The storage order of the voxels array is
    % such that this does not correspond with the global mm frame of reference.
    % Velocities from this algorithm are output in the local frame whose units
    % are voxels and whose values strictly increase with index into the arrays.
    % Conversion to the global frame is done by PIV_3d.m, the harness function
    % for this algorithm. Thus we only consider the local frame within this
    % code.
    %
    %       P#      Location (LOCAL Voxels frame)
    %        0      Xmin Ymin Zmin
    %        1      Xmin Ymax Zmin
    %        2      Xmax Ymax Zmin
    %        3      Xmax Ymin Zmin
    %        4      Xmin Ymin Zmax
    %        5      Xmin Ymax Zmax
    %        6      Xmax Ymax Zmax
    %        7      Xmax Ymin Zmax
    %
    % In this local voxels frame, the X direction is columnar, the Y direction
    % is row-wise, the Z direction is page-wise.
    ctrLoc = [wCtrsXGrid(:) wCtrsYGrid(:) wCtrsZGrid(:)];
    loc0 = ctrLoc + repmat( (wSize-1).*[-0.5 -0.5 -0.5], [nW 1]);
    loc1 = ctrLoc + repmat( (wSize-1).*[-0.5 +0.5 -0.5], [nW 1]);
    loc2 = ctrLoc + repmat( (wSize-1).*[+0.5 +0.5 -0.5], [nW 1]);
    loc3 = ctrLoc + repmat( (wSize-1).*[+0.5 -0.5 -0.5], [nW 1]);
    loc4 = ctrLoc + repmat( (wSize-1).*[-0.5 -0.5 +0.5], [nW 1]);
    loc5 = ctrLoc + repmat( (wSize-1).*[-0.5 +0.5 +0.5], [nW 1]);
    loc6 = ctrLoc + repmat( (wSize-1).*[+0.5 +0.5 +0.5], [nW 1]);
    loc7 = ctrLoc + repmat( (wSize-1).*[+0.5 -0.5 +0.5], [nW 1]);
    
    
    %% CONVECT WINDOW CORNER LOCATIONS ACCORDING TO INPUT VELOCITY FIELD
    
    % If there is no guess from a previous pass, then use first guess from
    % pivOpts (otherwise 0)
    if iPass == 1
        % If an input first guess is supplied...
        if isstruct(pivOpts.firstGuessField)
            guessField = pivOpts.firstGuessField;
        else
            guessField = 0;
        end
    end
    
    % Convect window corner locations according to the guess field,
    % which must have been converted to the units voxels/second. This routine
    % also checks for out of bounds errors, and reduces translation to 0 in the
    % event of Out Of Bounds.
    
    % Get the convected window corners, replicated for fields A, B etc
    [convLoc0 convLoc1 convLoc2 convLoc3,...
     convLoc4 convLoc5 convLoc6 convLoc7,...
            wConvection] = windowCornerConvect(loc0, loc1, loc2, loc3,...
                                               loc4, loc5, loc6, loc7,...
                                               guessField, dt, order,...
                                               [5 nvox_X-4], ...
                                               [5 nvox_Y-4], ...
                                               [5 nvox_Z-4]);
    disp(['    Window corner setup complete. Elapsed time ' num2str(toc) 's'])
    
    
    %% PERFORM CROSS CORRELATION AND PEAK LOCATION
        
    % Parallel flag
    if strcmpi(pivOpts.algorithm,'fmexpar') || strcmpi(pivOpts.algorithm,'fmexpar_cte')
        parFlag = true;
    else
        parFlag = false;
    end
    
    % Call the mex function wrapper with the two reconstructed intensity 
    % fields. Note the ordering; ur, uc, up translates to uy, ux, uz
    if strcmpi(order,'first')
        
        [dY dX dZ snr] = vodim3d(fieldA, fieldB, convLoc0, convLoc1, convLoc2, ...
                                     convLoc3, convLoc4, convLoc5, ...
                                     convLoc6, convLoc7, wSize(1), ...
                                     pivOpts.iwMaxDisp, ...
                                     pivOpts.fetchType(iPass), parFlag);
    else
%         [dY dX dZ snr vdY vdX vdZ vsnr ] = cte3d(...
%                                      fieldA, fieldB, fieldC, fieldD, ...
%                                      convLoc0, convLoc1, convLoc2, ...
%                                      convLoc3, convLoc4, convLoc5, ...
%                                      convLoc6, convLoc7, wSize(1), ...
%                                      pivOpts.iwMaxDisp, ...
%                                      pivOpts.fetchType(iPass), parFlag);
        [dY dX dZ snr] = cte3d(  fieldA, fieldB, fieldC, fieldD, ...
                                 convLoc0, convLoc1, convLoc2, ...
                                 convLoc3, convLoc4, convLoc5, ...
                                 convLoc6, convLoc7, wSize(1), ...
                                 pivOpts.iwMaxDisp, ...
                                 pivOpts.fetchType(iPass), parFlag);

    end
    
    disp(['    Successfully executed MEX file. Elapsed time ' num2str(toc) 's'])
    
    % Number of peaks fitted within each volume (this syntax allows for routines
    % which detect secondary or tertiary peaks in the cross correlation volumes)
    nPeaks = size(dX,2);
    
    % Add the window displacement (in voxels) to the computed displacement fields
    dX = dX + repmat(wConvection(:,1), [1 nPeaks]);
    dY = dY + repmat(wConvection(:,2), [1 nPeaks]);
    dZ = dZ + repmat(wConvection(:,3), [1 nPeaks]);
    
%     if iPass > 1
%     raiseFigure('Debugging plot of window convection');
%     clf
%     quiver3(wCtrsXGrid(:), wCtrsYGrid(:), wCtrsZGrid(:), wConvection(:,1), wConvection(:,2), wConvection(:,3))
%     end
    
    
    %% VALIDATE VECTOR FIELDS
    
    % Cross-correlation is complete for the current pass. Perform vector
    % validation on the current field:
    [valDX valDY valDZ valSnr medResidual] = PIV_3d_validatefield(nWindows, wCtrsXGrid, wCtrsYGrid, wCtrsZGrid, dX, dY, dZ, snr, iPass, pivOpts);
    
    % Gap fill where NaNs remain in the velocity arrays.
    [valDX valDY valDZ nanMask] = PIV_3d_gapfill(wCtrsXGrid, wCtrsYGrid, wCtrsZGrid, valDX, valDY, valDZ, pivOpts.gapFillMethod);
    
    % Occasionaly, for extremly poorly conditioned problems (or ones where the
    % inputs are just plain wrong) the vector field validation produces no valid
    % vectors. In this case, it causes misleading errors to attempt to continue
    % with the multipass algorithm. Throw an error here.
    if any( isnan(valDX(:)) | isnan(valDY(:)) | isnan(valDZ(:)) )
        error('MATLAB:TomoPIVToolbox:InvalidInput','Gap filling poorly conditioned, as 100% of the output velocities are invalid. Check inputs and try again.') %#ok<CTPCT>
    end
    
    % Store results in TomoPIV Toolbox format
    velocityField(iPass).ux = valDX/dt; % NB UNITS VOXELS / SECOND
    velocityField(iPass).uy = valDY/dt;
    velocityField(iPass).uz = valDZ/dt;
    velocityField(iPass).nanMask = nanMask;
    velocityField(iPass).winCtrsX = wCtrsXGrid;
    velocityField(iPass).winCtrsY = wCtrsYGrid;
    velocityField(iPass).winCtrsZ = wCtrsZGrid;
    velocityField(iPass).distanceUnits = 'voxels';
    velocityField(iPass).snr = valSnr;
    velocityField(iPass).medResidual = medResidual;
    
    % Save raw results for revalidation and investigation if requested
    if pivOpts.saveRaw
        % NB these fields always stored with distance units in voxels
        velocityField(iPass).rawdx_vox  = dX;
        velocityField(iPass).rawdy_vox  = dY;
        velocityField(iPass).rawdz_vox  = dZ;
        velocityField(iPass).rawsnr     = snr;
    end
    
    % If not the final pass, we use a smoothed version of the result for the
    % next pass (smoothing reduces noise and erroneous vectors). 
    if iPass < pivOpts.nPasses
        
        % Make a copy of the field to smooth and use as the next pass guess
        guessField = velocityField(iPass);
        
        % Determine method used to smooth
        if ischar(pivOpts.interPassFilter)
            interPassFilter = pivOpts.interPassFilter;
        else % it's a cell array of strings
            interPassFilter = pivOpts.interPassFilter{iPass};
        end
        
        % Switch according to method, and smooth the guess field
        switch lower(interPassFilter)
            case 'smooth3'
                guessField.ux = smooth3(guessField.ux,'gaussian',[3 3 3],3);
                guessField.uy = smooth3(guessField.uy,'gaussian',[3 3 3],3);
                guessField.uz = smooth3(guessField.uz,'gaussian',[3 3 3],3);
            case 'p4'
                [fux fuy fuz] = P4(guessField.ux, guessField.uy, guessField.uz);
                guessField.ux = fux;
                guessField.uy = fuy;
                guessField.uz = fuz;
            case 'relax34'
                
                % Call RELAX_34
                delta = abs(winCtrsX(2) -winCtrsX(1));
                [uxPOCS uyPOCS uzPOCS] = unirelax(...
                         guessField.ux, guessField.uy, guessField.uz, delta, ...
                                    'relaxation',           1,...
                                    'max_iters',            50,... 
                                    'break_conv',           false,...
                                    'zeroPadP3',            true,...
                                    'debug',                false,...
                                    'p4relaxationfactor',   1/50);
                guessField.ux = uxPOCS;
                guessField.uy = uyPOCS;
                guessField.uz = uzPOCS;

            otherwise
                error('MATLAB:TomoPIVToolbox:InvalidInput','interPassFilter field in the pivOpts structure has an invalid format. Valid options are ''smooth3'',''p4'',''relax34'' or a cell array consisting of one of those two strings for each pass')
        end
    end
    
%     % If CTE is used, then we also validate the VODIM results
%     if pivOpts.saveVODIM && ~strcmpi(order,'first')
% 
%         % Add the window displacement (in voxels) to the computed displacement fields
%         vdX = vdX + repmat(wConvection(:,1), [1 nPeaks]);
%         vdY = vdY + repmat(wConvection(:,2), [1 nPeaks]);
%         vdZ = vdZ + repmat(wConvection(:,3), [1 nPeaks]);
%         
%         % VALIDATE VECTOR FIELDS
%     
%         % Cross-correlation is complete for the current pass. Perform vector
%         % validation on the current field:
%         [valDX valDY valDZ valSnr medResidual] = PIV_3d_validatefield(nWindows, wCtrsXGrid, wCtrsYGrid, wCtrsZGrid, vdX, vdY, vdZ, vsnr, iPass, pivOpts);
%     
%         % Gap fill where NaNs remain in the velocity arrays.
%         [valDX valDY valDZ nanMask] = PIV_3d_gapfill(wCtrsXGrid, wCtrsYGrid, wCtrsZGrid, valDX, valDY, valDZ, pivOpts.gapFillMethod);
%     
%         % Occasionaly, for extremly poorly conditioned problems (or ones where the
%         % inputs are just plain wrong) the vector field validation produces no valid
%         % vectors. In this case, it causes misleading errors to attempt to continue
%         % with the multipass algorithm. We don't error, as it's quite
%         % possible that the CTE algorithm is computing good results, but we
%         % do warn and conditionally process
%         if any( isnan(valDX(:)) | isnan(valDY(:)) | isnan(valDZ(:)) )
%             warning('MATLAB:TomoPIVToolbox:InvalidInput','VODIM Gap filling poorly conditioned, as 100% of the output velocities are invalid. This will not affect results of the CTE algorithm') %#ok<CTPCT>
%         end
%     
%         % Store results in TomoPIV Toolbox format. It's on the same grid so
%         % no need to store the grid data
%         vodimStruct.ux = valDX/dt; % NB UNITS VOXELS / SECOND
%         vodimStruct.uy = valDY/dt;
%         vodimStruct.uz = valDZ/dt;
%         vodimStruct.snr = valSnr;
%         vodimStruct.nanMask = nanMask;
%         vodimStruct.distanceUnits = 'voxels';
%         vodimStruct.medResidual = medResidual;
%     
%         % Save raw results for revalidation and investigation if requested
%         if pivOpts.saveRaw
%             % NB these fields always stored with distance units in voxels
%             vodimStruct.rawdx_vox  = vdX;
%             vodimStruct.rawdy_vox  = vdY;
%             vodimStruct.rawdz_vox  = vdZ;
%             vodimStruct.rawsnr     = vsnr;
%         end
%         
%         % Insert into the velocityStruct for the CTE results (that way we
%         % don't have to do any special handling of the load/save commands)
%         velocityField(iPass).vodimStruct = vodimStruct;
%     end
    
    % Display the tictoc
    disp(['    Completed pass. Elapsed time ' num2str(toc) 's'])
    disp('_______________________________________________________________')
    disp(' ')
    
end % end for iPass






end % END FUNCTION PIV_3d_cmex
