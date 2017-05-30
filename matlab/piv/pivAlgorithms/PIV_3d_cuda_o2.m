function [velocityStruct] = PIV_3d_cuda_o2(fieldA, fieldB, fieldC, fieldD, dt, pivOpts)
%PIV_3d Particle Image Velocimetry using 3D scalar intensity (particle) fields
%   Performs windowed cross correlation using second oreder time-accurate
%   algorithm
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
%       [] = PIV_3d_cuda_o2(field1, field2, dx, dt, pivOpts)
%            Cross-correlates field1 (particle field at time t) with field 2
%            (particle field at time t + dt)
%
%
%       fieldA, fieldB, fieldC, fieldD
%                               [nVoxY x nVoxZ x nVoxZ] single
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
%   [1] Raffel M. Willert C. Wereley S. and Kompenhans J. Particle Image
%       Velocimetry A Practical Guide (Second Edition). Springer 2007 
%       ISBN 978-3-540-72307-3
%
%   [2] Thomas M. Misra S. Kambhamettu C. and Kirby J.T. (2005)
%       A robust motion estimation algorithm for PIV
%
% Future Improvements:
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
% Revison History:          15 April 2010       Created
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
nvox_X = size(field1,2);
nvox_Y = size(field1,1);
nvox_Z = size(field1,3);

