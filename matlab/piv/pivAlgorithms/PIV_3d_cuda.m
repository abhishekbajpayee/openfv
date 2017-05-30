function [velocityStruct] = PIV_3d_cuda(fieldA, fieldB, dt, pivOpts)
%PIV_3d Particle Image Velocimetry using 3D scalar intensity (particle) fields
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
%   previous pass as a first guess, and a window deformation algorithm is used.
%
% Syntax:  
%       [] = PIV_3d_cuda(fieldA, fieldB, dt, pivOpts)
%            Cross-correlates fieldA (particle field at time t) with field 2
%            (particle field at time t + dt)
%
%
%       fieldA, fieldB          [nVoxY x nVoxZ x nVoxZ] single
%                                               Contains the reconstructed
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
%       velocityStructure       structure(nPasses)
%                                               A structure containing the
%                                               intermediate and final velocity
%                                               fields produced by PIV_3d. This
%                                               structure can be used in further
%                                               plotting, analysis and
%                                               prostprocessing of the results.
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
%       intensity fields to the GPU at each step.
%
%   [2] 
%       
% Other m-files required:   none
% Subfunctions:             weightingMatrix Computes the 3D cross correlation
%                           weighting matrix (see ref. 1)
% Nested functions:         none
% MAT-files required:       none
%
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
%                           01 May 2011         Minor doc change

%   Copyright (c) 2007-2015  Thomas H. Clark



%% PRELIMINARY CALCULATIONS AND SETUP

% Initialise the results structure
emptycell      = cell(pivOpts.npasses,1);
velocityStruct = struct('ux',           emptycell, ...
                        'uy',           emptycell, ...
                        'uz',           emptycell, ...
                        'indicator',    emptycell, ...
                        'peak_locs',    emptycell, ...
                        'peak_vals',    emptycell, ...
                        'peak_void',    emptycell, ...
                        'win_ctrs_x',   emptycell, ...
                        'win_ctrs_y',   emptycell, ...
                        'win_ctrs_z',   emptycell);
   
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
    
    % Check for out-of-bounds errors
    
    % Where out-of bounds errors occur, decrease the translation by half.
    % Recalculate window corder locations, check again.
    
    % Where out-of-bounds errors occur, decrease the translation to zero.
    % Recalculate window corner locations, check again.
    
    % Where out-of-bounds errors occur, eliminate the windows from the
    % cross-correlation
    
    % Call cudaCrossCorr with the four velocity fields
    
    % Perform peak location on the output windows (?)
    
    % Perform vector validation
    
    % Store results in TomoPIV Toolbox format
    
    % If not the final pass, smooth the velocity field ready for next pass
    
% end




% CONVERT FIRST GUESS TO A VELOCITY FIELD





% LOOP FOR EACH PASS
for iPass = 1:pivOpts.nPasses

    
    % Display progress
    disp(['PIV_3d_matlab.m: Pass ' num2str(iPass) ' of ' num2str(pivOpts.nPasses)])
    
    
    
    %% GET PARAMETERS FOR THIS PASS
    
    % Window sizes for current pass...
    wSize = pivOpts.wSize(iPass,:);
    if size(wSize,2) == 1
        wSize = [wSize wSize wSize]; %#ok<AGROW>
    elseif isequal(wSize(1),wSize(2),wSize(3))
        error('MATLAB:TomoPIVToolbox:InvalidInput','For CUDA based PIV algorithms, window size must be the same in all directions (cuboid windows)')
    end
    
    % Overlap
    overlap = pivOpts.overlap(iPass);
    if (overlap < 0) || (overlap >= 100)
        error('MATLAB:TomoPIVToolbox:InvalidInput','Overlap cannot be < 0 or >= 100 %')
    end
    
    % Edge cut (number of voxels to shave off all faces of the volume)
    edgeCut = pivOpts.edgeCut;
    
        
    

    %% DETERMINE WINDOW POSITIONS FOR CURRENT PASS
    
    % Max and min locations (in local voxel coords)
    minCoord =                          wSize/2 + 0.5 + edgeCut;
    maxCoord = [nvox_X nvox_Y nvox_Z] - wSize/2 + 0.5 - edgeCut;
    
    % Window spacing in voxels in the X,Y,Z directions
    wSpacing = wSize * (1 - (overlap/100));
    
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
    
    % Form 3D meshes of these coordinates and convert to list form
    [wCtrsXGrid wCtrsYGrid wCtrsZGrid] = ndgrid(wCtrsX,wCtrsY,wCtrsZ);
    wCtrsXGrid = wCtrsXGrid(:);
    wCtrsYGrid = wCtrsYGrid(:);
    wCtrsZGrid = wCtrsZGrid(:);
    
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
    % teh indexing of the voxels array. The storage order of the voxels array is
    % such that this does not correspond with the global mm frame of reference.
    % Velocities from this algorithm are output in the local frame, and are
    % converted to the global frame by PIV_3d.m, the harness function for this
    % algorithm. Thus we only consider the local fram within this code.
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
    ctrLoc = [wCtrsXGrid wCtrsYGrid wCtrsZGrid];
    loc0 = ctrLoc + repmat( wSize.*[-0.5 -0.5 -0.5], [nW 1]);
    loc1 = ctrLoc + repmat( wSize.*[-0.5 +0.5 -0.5], [nW 1]);
    loc2 = ctrLoc + repmat( wSize.*[+0.5 +0.5 -0.5], [nW 1]);
    loc3 = ctrLoc + repmat( wSize.*[+0.5 -0.5 -0.5], [nW 1]);
    loc4 = ctrLoc + repmat( wSize.*[-0.5 -0.5 +0.5], [nW 1]);
    loc5 = ctrLoc + repmat( wSize.*[-0.5 +0.5 +0.5], [nW 1]);
    loc6 = ctrLoc + repmat( wSize.*[+0.5 +0.5 +0.5], [nW 1]);
    loc7 = ctrLoc + repmat( wSize.*[+0.5 -0.5 +0.5], [nW 1]);
    
    
    
    %% CONVECT WINDOW CORNER LOCATIONS ACCORDING TO INPUT VELOCITY FIELD
    
    % Convect window corner locations according to the input velocity field,
    % which must have been converted to the units voxels/second. This routine
    % also checks for out of bounds errors, and reduces translation to 0 in the
    % event of OOB.
    % Returned values are arrays of size [nW x 3 x 2]. The first dimension
    % corresponds to the number of windows. The three columns are for X,Y,Z 
    % locations of the window corners. The third dimension (pages) represents
    % the number of volumes. So for this case, cross correlating A and B, then
    % convLoc#(:,:,1) represents the window corner indices to be fetched from
    % field A, and convLoc#(:,:,2) represent window corner indices to be fetched
    % from field B. This technique is extensible to four fields for a second
    % order computation.
    order = 'first';
    [convLoc0 convLoc1 convLoc2 convLoc3,...
     convLoc4 convLoc5 convLoc6 convLoc7,...
        wConvection] = windowCornerConvect(loc0, loc1, loc2, loc3,...
                                           loc4, loc5, loc6, loc7,...
                                           guessVelocityArray, dt, order);
    
                                       
                                       
    %% PERFORM CROSS CORRELATION AND PEAK LOCATION
                                       
    % Call cudaCrossCorr with the two reconstructed intensity fields
    [dX dY dZ] = cudaPIV(fieldA, fieldB, whatIsTheSyntacticarrangemnt);
        
    % Number of peaks fitted within each volume (this syntax allows for routines
    % which detect secondary or tertiary peaks in the cross correlation volumes)
    nPeaks = size(dX,3);
    
    % Add the window displacement to the computed displacement fields
    dX = dX + repmat(wConvection(:,1), [1 1 nPeaks]);
    dY = dY + repmat(wConvection(:,2), [1 1 nPeaks]);
    dZ = dZ + repmat(wConvection(:,3), [1 1 nPeaks]);
    
    
    
    %% VALIDATE VECTOR FIELDS
    
    % Cross-correlation is complete for the current pass. Perform vector
    % validation on the current field:
    [valDX valDY valDZ] = validateField(nWindows, dX, dY, dZ, iPass, pivOpts);
    
    % Gap fill where NaNs remain in the velocity arrays.
    [valDX valDY valDZ nanMask] = PIV_3d_gapfill(mesh_x, mesh_y, mesh_z, valDX, valDY, valDZ, pivOpts.gapFillMethod);
    
    % Occasionaly, for extremly poorly conditioned prblems (or ones where the
    % inputs are just plain wrong) the vector field validation produces no valid
    % vectors. In this case, it causes misleading errors to attempt to continue
    % with the multipass algorithm. Throw an error here.
    if any( isnan(dX(:)) | isnan(dY(:)) | isnan(dZ(:)) )
        error('MATLAB:TomoPIVToolbox:InvalidInput','Gap filling poorly conditioned, as 100% output velocities are invalid. Check inputs and try again.')
    end
    
    % Store results in TomoPIV Toolbox format
    
    % If not the final pass, smooth the velocity field ready for next pass
    if iPass < pivOpts.nPasses
        %% NOTE TO TOM - REFORMULATE THIS AS inputGessField = [blah blah blah];
        ux = smooth3(ux,'gaussian',[3 3 3],3);
        uy = smooth3(uy,'gaussian',[3 3 3],3);
        uz = smooth3(uz,'gaussian',[3 3 3],3);
    end
    
    
end % end for iPass


    
        
    % Store the results of this pass to a structure.
    velocityStruct(pass).ux = ux/dt;
    velocityStruct(pass).uy = uy/dt;
    velocityStruct(pass).uz = uz/dt;
    velocityStruct(pass).nan_mask = nan_mask;
    velocityStruct(pass).pk_locs_x = pk_locs_x;
    velocityStruct(pass).pk_locs_y = pk_locs_y;
    velocityStruct(pass).pk_locs_z = pk_locs_z;
    velocityStruct(pass).win_ctrs_x = win_ctrs_x;
    velocityStruct(pass).win_ctrs_y = win_ctrs_y;
    velocityStruct(pass).win_ctrs_z = win_ctrs_z;
    

end % End for pass = 1:npasses






end % END FUNCTION PIV_3d_matlab
