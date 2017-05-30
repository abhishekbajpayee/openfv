function [velocityStruct] = PIV_3d(varargin)
%PIV_3d Particle Image Velocimetry using 3D scalar intensity (particle) fields
% Windowed cross correlation using FFT approach for 3D3C measurement of velocity
% 
% PIV_3d serves as a harness for several different PIV algorithms, ensuring
% compatibility with the TomoPIV Toolbox.
% 
% The following algorithms are supported:
%
%       [1] PIV_3d_matlab
%           Performs the cross correlation in MATLAB. This is extremely slow,
%           but works well and does not require precompilation (i.e. good for
%           first use of the TomoPIV Toolbox). Features include window shifting
%           (not deformation) and a multiple pass option. A wide range of false
%           vector identification criteria are available, including the
%           normalised median test, RMS differences, turbulent intensities and
%           maximum values. Peak location algorithm can be specified (either
%           Gaussian fit or Whittaker/Sinc interpolation).
%
%       [2] PIV_3d_matlab_cte
%           Second order accurate equivalent of PIV_3d_matlab. Note that
%           multipass behaviour is not supported for this test/demonstration
%           algorithm.
%
%       [3] PIV_3d_fMex
%           Performs the cross correlation using FORTRAN MEX file.
%           Features include window deformation and multi-pass option.
%           A wide range of false vector identification criteria are 
%           available, including the normalised median test, 
%           RMS differences, turbulent intensities and maximum values.
%           Gaussian peak location is used. MEX file is parallelised using
%           OpenMP.
%
%       [4] PIV_3d_fMex_cte
%           CTE algorithm based on PIV_3d_fMex.
%
%       [5] PIV_3d_nawcc
%           First order cross correlation with window shifting (not deformation)
%           and multipass capability. This is coded in FORTRAN with a MEX
%           interface to MATLAB. Thus this can be run on systems without CUDA
%           capability. The only false vector validation available is RMS
%           difference. Due to a legacy limitation, the file must be recompiled
%           for different window sizes. Recompilation required the Intel Fortran
%           Compiler and Intel Math Kernel Library.
%           This code has been kindly supplied by Nick Worth (from the 'TomoPRO'
%           code which Nick developed), and modified to add a MATLAB interface
%           by Tom Clark
%
%       [6] PIV_3d_cuda
%           First order cross correlation with full window deformation and
%           multipass capability. The core window deformation and FFT algorithms
%           are written in C Runtime API for CUDA, hence are massively
%           parallelised. This algorithm has a system requirement of an 
%           NVIDIA graphics cards. Current memory requirement for the graphics
%           card is ~4Gb although development work is ongoing to decrease the
%           memory requirements. The core algorithms are harnessed by MATLAB so
%           the same false vector selection algorithms are available as for
%           PIV_3d_matlab.
%
%       [7] PIV_3d_cuda_o2
%           Second order accurate equivalent of PIV_3d_cudaBatchCorr.
%
% Syntax:  
%
%       velocityStructure = PIV_3d(fieldA, fieldB, setup, dt, pivOpts)
%   
%            Cross-correlates fieldA (particle field at time t) with field B
%            (particle field at time t + dt) and returns results in a velocity
%            structure. Run-time options are specified using the pivOpts
%            structure (see below).
%
%       velocityStructure = PIV_3d(fieldA, fieldB, fieldC, fieldD, setup, dt, pivOpts)
%
%           Performs CTE based cross correlation of the
%           fields, using the formula crossCorr = (AxB + 2*BxC + CxD)/4
%
%       
% Inputs:
%    
%       fieldA, fieldB, fieldC, fieldD
%                               [nVox x 1] single or
%                               [nVoxY x nVoxZ x nVoxZ] single
%                                               Contains the reconstructed
%                                               intensity field at a point in
%                                               time, where A,B,C,D represent a
%                                               consecutive time series, each
%                                               frame separated by time dt. See
%                                               the reconstruction functions for
%                                               details on the storage order and
%                                               shape of the reconstruction
%                                               fields.
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
% References:
%
%   [1] Raffel M. Willert C. Wereley S. and Kompenhans J. Particle Image
%       Velocimetry A Practical Guide (Second Edition). Springer 2007 
%       ISBN 978-3-540-72307-3
%
%   [2] Thomas M. Misra S. Kambhamettu C. and Kirby J.T. (2005)
%       A robust motion estimation algorithm for PIV
%
% Future Improvements:      none
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
% Revision History:         15 September 2009   Created
%                           08 July 2010        Reformulated and some work
%                           16 April 2011       Completely revised documentation
%                                               and format of function. Original
%                                               functionality cut out and placed
%                                               as harnessed function
%                                               PIV_3d_matlab. PIV_3d is
%                                               now reformulated to call any of
%                                               the PIV algorithms (as specified
%                                               in the pivOpts structure), so
%                                               that a user can always call
%                                               PIV_3d, but simply alter the
%                                               specified options in the pivOpts
%                                               structure to change which PIV
%                                               algorithm is used. This makes
%                                               the TomoPIV Toolbox usage
%                                               scripts much clearer and makes
%                                               it much easier to add
%                                               alternative PIV algorithms
%                           15 August 2011      Altered doc to reflect
%                                               parallelised fMex
%                                               algorithms and change O2 to
%                                               CTE. Fixed a few typos.
%                           31 August 2011      Added a warning in the
%                                               event that the user tries
%                                               an old field format in the
%                                               PIV options structure

%   Copyright (c) 2007-2015  Thomas H. Clark

%% COMMON CHECKS ON INPUTS

% Get setup
setup = varargin{end-2};

% Check that dt is > 0
dt = varargin{end-1};
if dt <= 0
    error('MATLAB:TomoPIVToolbox:InvalidInput', 'Value of dt cannot be <= 0. Input the time in seconds between successive image frames')
end

% Check that pivOpts is a stucture
pivOpts = varargin{end};
if ~isstruct(pivOpts)
    error('MATLAB:TomoPIVToolbox:InvalidInput', 'Input pivOpts must be a PIV options structure as defined by the function definePIVOptions()')
end

% Get the sizes of the reconstruction arrays (Used for reshaping the input
% arrays if necessary).
nX = numel(setup.vox_X);
nY = numel(setup.vox_Y);
nZ = numel(setup.vox_Z);



%% CUSTOM CHECKS FOR DIFFERENT ALGORITHMS

switch lower(pivOpts.algorithm)
    
    case 'matlab'
        
        % Check for the right number of input arguments
        if nargin ~= 5
            error('MATLAB:TomoPIVToolbox:IncorrectNumberOfArguments','The MATLAB algorithm has been specified, but there is an incorrect number of input arguments for use with a second order algorithm. Call ''PIV_3d(fieldA, fieldB, setup, dt, pivOpts)''')
        end
        
        % Check the first four arguments are the same size and shape
        if ~isequal(size(varargin{1}), size(varargin{2}))
            error('MATLAB:TomoPIVToolbox:InconsistentArgumentSize','The first two arguments (fieldA, fieldB) must be of equal size and shape')
        end
        
    case 'matlab_cte'
        
        % Check for the right number of input arguments
        if nargin ~= 7
            error('MATLAB:TomoPIVToolbox:IncorrectNumberOfArguments','The 2nd order MATLAB algorithm has been specified, but there is an incorrect number of input arguments for use with a second order algorithm. Call ''PIV_3d(fieldA, fieldB, fieldC, fieldD, setup, dt, pivOpts)''')
        end
        
        % Check the first four arguments are the same size and shape
        if ~isequal(size(varargin{1}), size(varargin{2}), size(varargin{3}), size(varargin{4}))
            error('MATLAB:TomoPIVToolbox:InconsistentArgumentSize','The first four arguments (fieldA, fieldB, fieldC and fieldD) must be of equal size and shape')
        end
        
        % Check that the number of passes doesn't exceed 1
        if pivOpts.npasses > 1
            error('MATLAB:TomoPIVToolbox:ExceededMaxNumberOfPasses','The MATLAB implementation of the second order accurate PIV algorithm does not allow for multiple passes (unresolved bug in the code). Use this algorithm as single pass for test purposes only.')
        end
        
    case 'fmex'
        
        % Check for the right number of input arguments
        if nargin ~= 5
            error('MATLAB:TomoPIVToolbox:IncorrectNumberOfArguments','The fMex algorithm has been specified, but there is an incorrect number of input arguments for use with a second order algorithm. Call ''PIV_3d(fieldA, fieldB, setup, dt, pivOpts)''')
        end
        
        % Check the first four arguments are the same size and shape
        if ~isequal(size(varargin{1}), size(varargin{2}))
            error('MATLAB:TomoPIVToolbox:InconsistentArgumentSize','The first two arguments (fieldA, fieldB) must be of equal size and shape')
        end
        
    case 'fmexpar'
        
        % Check for the right number of input arguments
        if nargin ~= 5
            error('MATLAB:TomoPIVToolbox:IncorrectNumberOfArguments','The fMexpar algorithm has been specified, but there is an incorrect number of input arguments for use with a second order algorithm. Call ''PIV_3d(fieldA, fieldB, setup, dt, pivOpts)''')
        end
        
        % Check the first four arguments are the same size and shape
        if ~isequal(size(varargin{1}), size(varargin{2}))
            error('MATLAB:TomoPIVToolbox:InconsistentArgumentSize','The first two arguments (fieldA, fieldB) must be of equal size and shape')
        end
        
    case 'fmex_cte'
        
        % Check for the right number of input arguments
        if nargin ~= 7
            error('MATLAB:TomoPIVToolbox:IncorrectNumberOfArguments','The fMex_cte algorithm has been specified, but there is an incorrect number of input arguments for use with a second order algorithm. Call ''PIV_3d(fieldA, fieldB, fieldC, fieldD, setup, dt, pivOpts)''')
        end
        
        % Check the first four arguments are the same size and shape
        if ~isequal(size(varargin{1}), size(varargin{2}), size(varargin{3}), size(varargin{4}))
            error('MATLAB:TomoPIVToolbox:InconsistentArgumentSize','The first four arguments (fieldA, fieldB, fieldC and fieldD) must be of equal size and shape')
        end
        
    case 'fmexpar_cte'
        
        % Check for the right number of input arguments
        if nargin ~= 7
            error('MATLAB:TomoPIVToolbox:IncorrectNumberOfArguments','The fMexpar_cte algorithm has been specified, but there is an incorrect number of input arguments for use with a second order algorithm. Call ''PIV_3d(fieldA, fieldB, fieldC, fieldD, setup, dt, pivOpts)''')
        end
        
        % Check the first four arguments are the same size and shape
        if ~isequal(size(varargin{1}), size(varargin{2}), size(varargin{3}), size(varargin{4}))
            error('MATLAB:TomoPIVToolbox:InconsistentArgumentSize','The first four arguments (fieldA, fieldB, fieldC and fieldD) must be of equal size and shape')
        end
        
    case 'cuda'
        
        % PIV_3d_cuda is only partly implmented and hasn't been validated yet.
        % Throw an error, rather than risk getting things wrong.
        error('MATLAB:TomoPIVToolbox:NotImplementedYet', 'Functionality of PIV_3d_cuda only partly implemented - under development')
        
    case 'cuda_cte'
        
        % PIV_3d_cuda_o2 is only partly implmented and hasn't been validated yet.
        % Throw an error, rather than risk getting things wrong.
        error('MATLAB:TomoPIVToolbox:NotImplementedYet', 'Functionality of PIV_3d_cuda_o2 only partly implemented - under development')
        
    case 'naw'
        
        % nawCrossCorr is only partly implmented and hasn't been validated yet.
        % Throw an error, rather than risk getting things wrong.
        error('MATLAB:TomoPIVToolbox:NotImplementedYet', 'Functionality of nawCrossCorr only partly implemented - under development')
    
    otherwise
        % nawCrossCorr is only partly implmented and hasn't been validated yet.
        % Throw an error, rather than risk getting things wrong.
        error('MATLAB:TomoPIVToolbox:InvalidInput', 'The algorithm specified in the pivOpts structure is invalid. Try ''fMex'' or ''fmex_cte''.')
        
end




%% CALL PIV ALGORITHM

switch lower(pivOpts.algorithm)
    
    case 'matlab'
        
        % Call the first order accurate MATLAB-Based PIV algorithm
        velocityStruct = PIV_3d_matlab(     reshape(varargin{1},nY,nX,nZ),...
                                            reshape(varargin{2},nY,nX,nZ),...
                                            dt, pivOpts);
        
    case 'matlab_cte'
        
        % Call the second order accurate MATLAB-Based PIV algorithm
        velocityStruct = PIV_3d_matlab_o2(  reshape(varargin{1},nY,nX,nZ),...
                                            reshape(varargin{2},nY,nX,nZ),...
                                            reshape(varargin{3},nY,nX,nZ),...
                                            reshape(varargin{4},nY,nX,nZ),...
                                            dt, pivOpts);
        
        
    case 'fmex'
        
        % Call the first order accurate CUDA-Based PIV algorithm
        velocityStruct = PIV_3d_mex (       reshape(varargin{1},nY,nX,nZ),...
                                            reshape(varargin{2},nY,nX,nZ),...
                                            dt, pivOpts);
                                        
    case 'fmexpar'
        
        % Call the first order accurate CUDA-Based PIV algorithm
        velocityStruct = PIV_3d_mex (       reshape(varargin{1},nY,nX,nZ),...
                                            reshape(varargin{2},nY,nX,nZ),...
                                            dt, pivOpts);
        
    case 'fmex_cte'
        
        % Call the second order accurate CUDA-Based PIV algorithm
        velocityStruct = PIV_3d_mex (       reshape(varargin{1},nY,nX,nZ),...
                                            reshape(varargin{2},nY,nX,nZ),...
                                            reshape(varargin{3},nY,nX,nZ),...
                                            reshape(varargin{4},nY,nX,nZ),...
                                            dt, pivOpts);
        
    case 'fmexpar_cte'
        
        % Call the second order accurate CUDA-Based PIV algorithm
        velocityStruct = PIV_3d_mex (       reshape(varargin{1},nY,nX,nZ),...
                                            reshape(varargin{2},nY,nX,nZ),...
                                            reshape(varargin{3},nY,nX,nZ),...
                                            reshape(varargin{4},nY,nX,nZ),...
                                            dt, pivOpts);
    case 'naw'
        
        % REQUIRES MODIFICATIONS - this code will error if run.
        velocityStruct = nawcc(             reshape(varargin{1},nY,nX,nZ),...
                                            reshape(varargin{2},nY,nX,nZ),...
                                            dt, pivOpts);
        

end



%% CONVERT VELOCITY STRUCTURE TO CONTAIN CORRECT UNITS AND COORDINATE FRAME

% Velocities in the returned structures are in units of voxels/second. The
% window centre locations (also in the velocities structures) are in voxel
% units, relative to the local coordinate frame (i.e. the reconstruction
% volume). The setup structure contains details of the voxel size in mm, and the
% position of the reconstruction region in global space. We use these details to
% update the velocity structure to units of mm/s (velocities) and mm in the
% global frame (window centre positions).

% Get voxel size in mm
voxSizeMM = abs(setup.vox_X(2) - setup.vox_X(1));

for iPass = 1:pivOpts.nPasses
    
    % Set recorded units to mm in the structure
    velocityStruct(iPass).distanceUnits = 'mm';
    
    % Store the voxel size in mm
    velocityStruct(iPass).voxSizeMM = voxSizeMM;
    
    % Store the dt with it
    velocityStruct(iPass).dt = dt;

    % TODO: Why that negative sign for uy and uz...wtf?
    %       Also, not sure about that voxSizeMM scaling
    % Convert velocities to mm/s from voxels/s
    % velocityStruct(iPass).ux =    voxSizeMM*velocityStruct(iPass).ux;
    % velocityStruct(iPass).uy = -1*voxSizeMM*velocityStruct(iPass).uy;
    % velocityStruct(iPass).uz = -1*voxSizeMM*velocityStruct(iPass).uz;

    velocityStruct(iPass).ux = velocityStruct(iPass).ux;
    velocityStruct(iPass).uy = velocityStruct(iPass).uy;
    velocityStruct(iPass).uz = velocityStruct(iPass).uz;

    
    % If the structure is a CTE one containing VODIM results...
    if isfield(velocityStruct,'vodimStruct')
        
        % Set recorded units to mm in the structure
        velocityStruct(iPass).vodimStruct.distanceUnits = 'mm';
        
        % Convert velocities to mm/s from voxels/s
        velocityStruct(iPass).vodimStruct.ux =    voxSizeMM*velocityStruct(iPass).vodimStruct.ux;
        velocityStruct(iPass).vodimStruct.uy = -1*voxSizeMM*velocityStruct(iPass).vodimStruct.uy;
        velocityStruct(iPass).vodimStruct.uz = -1*voxSizeMM*velocityStruct(iPass).vodimStruct.uz;
        
    end
    
    % Offset to zero based (window centres refer to the 1-based index of the
    % voxels) and convert to mm
    velocityStruct(iPass).winCtrsX = voxSizeMM*(velocityStruct(iPass).winCtrsX - 1);
    velocityStruct(iPass).winCtrsY = voxSizeMM*(velocityStruct(iPass).winCtrsY - 1);
    velocityStruct(iPass).winCtrsZ = voxSizeMM*(velocityStruct(iPass).winCtrsZ - 1);
    
    % Offset and orient to the global frame of reference. 
    % Note the global frame Z is positive toward the camera (in the direction of
    % decreasing page index) and Y is positive upwards (in the direction of
    % decreasing row index). Thus the window centres (which increase with
    % direction of increasing row, column and page) must be negated to align the
    % vectors in the global frame.
    % velocityStruct(iPass).winCtrsX =     velocityStruct(iPass).winCtrsX  + setup.vox_X(1);
    % velocityStruct(iPass).winCtrsY = (-1*velocityStruct(iPass).winCtrsY) + setup.vox_Y(1);
    % velocityStruct(iPass).winCtrsZ = (-1*velocityStruct(iPass).winCtrsZ) + setup.vox_Z(1);

    velocityStruct(iPass).winCtrsX = velocityStruct(iPass).winCtrsX + setup.vox_X(1);
    velocityStruct(iPass).winCtrsY = velocityStruct(iPass).winCtrsY + setup.vox_Y(1);
    velocityStruct(iPass).winCtrsZ = velocityStruct(iPass).winCtrsZ + setup.vox_Z(1);

    
end
    


%% STORE AND RETURN RESULTS

% Store the resulting vector field...
if ~isnumeric(pivOpts.storeResult)
    % Then full path and file name has been passed, save a file
    status = tomoPIV_savevelocity(velocityStruct,pivOpts.storeResult);
    
    % Throw an error if the process failed.
    if status == 1
        disp(['tomoPIV_savevelocity.m: Attempted to save file: ' pivOpts.storeResult])
        error('MATLAB:TomoPIVToolkit:FailedOperation','Velocity Structure File - save process failed')
    end
    
elseif isfield(pivOpts,'store_result')
    
    % I've had a few problems with this when loading old piv options files.
    % Drove me mad, so there's a warning, although this should never affect
    % new users
    warning('MATLAB:TomoPIVToolbox','PIV_3d.m did not save velocity file. PIV Options structure contains a deprecated ''store_result'' field which should be updated to ''storeResult'' in order to save correctly')    
end


end % END MAIN FUNCTION PIV_3d


