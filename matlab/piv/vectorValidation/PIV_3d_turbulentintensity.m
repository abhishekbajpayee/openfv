function [turb_test] = PIV_3d_turbulentintensity(u, v, w, turbIntensity, velocity)
%PIV_3D_TURBULENTINTENSITY Vector field validation using turbulent intensity
%   Determines which vectors in a field violate a limit of turbulent intensity.
%   This allows users to select false vectors based on the 
%   A 3x3 matrix is used to express the turbulent intensity limits as a
%   fraction of the input velocity vector.
%   In the special case where components of the input velocity vector = 0, 
%   turbulent intensity limits are taken as absolute values.
%
% Syntax:
%       [turb_test] = PIV_3d_turbulentintensity(u, v, w, turbIntensity, velocity)
%                       (Preferred usage) allows user to specify the mean
%                       velocity vector
%
% Inputs:
%       u       [ny, nx, nz] or [npoints, 1]
%                               Array containing x-components of velocity field.
%
%       v       [ny, nx, nz] or [npoints, 1]
%                               Array containing y-components of velocity field.   
%
%       w       [ny, nx, nz] or [npoints, 1]
%                               Array containing z-components of velocity field.
%
%       turbIntensity [3 x 3]   
%                               Array containing turbulent intensities. The
%                               format of the array gives the variation in 
%                               turbulent fluctuation velocity as a function of
%                               the input components U, V, W.
%                               so:
%                                         U    V    W
%                                      u' t11  t12  t13
%                                      v' t21  t22  t23
%                                      w' t31  t32  t33
%                               
%                               Provides the constraints:
%                                       -U*t11 <= u' <= U*t11
%                                       -V*t12 <= u' <= V*t12
%                                       -W*t13 <= u' <= W*t13
%
%                                       -U*t21 <= v' <= U*t21
%                                       -V*t22 <= v' <= V*t22
%                                       -W*t23 <= v' <= W*t23
%
%                                       -U*t31 <= w' <= U*t31
%                                       -V*t32 <= w' <= V*t32
%                                       -W*t33 <= w' <= W*t33
%
%                               Where:
%                                       u' = u - U
%                                       v' = v - V
%                                       w' = w - W
%
%                               Except in the case where input components U, V
%                               or W are set = 0. In that case, turbulent
%                               intensities in the corresponding column are
%                               considered absolute. so, for the case where
%                               input velocity UVW = [10 0 0], the following
%                               constraints are applied:
%
%                                       -U*t11 <= u' <= U*t11
%                                         -t12 <= u' <= t12
%                                         -t13 <= u' <= t13
%
%                                       -U*t21 <= v' <= U*t21
%                                         -t22 <= v' <= t22
%                                         -t23 <= v' <= t23
%
%                                       -U*t31 <= w' <= U*t31
%                                         -t32 <= w' <= t32
%                                         -t33 <= w' <= t33
%                               
%                               Note that there is some redundancy in that case
%                               (i.e.  -min(t12,t13) <= u' <= min(t12,t13) ) so 
%                               be careful when specifying absolute turbulent 
%                               intensities.
%
%                               Set tij = Inf to remove a particular
%                               constraint.
%   
%       velocity      [1 x 3] or [3 x 1]
%                               Vector containing U, V, W components. These are
%                               typically mean values or idealised mean values
%                               in a field. For example, in a boundary layer,
%                               this input may be [mean(u) 0 0], representing a
%                               mean flow in the x direction, with no flow in 
%                               the cross stream (y) and wall-normal (z) 
%                               directions.
%
% Outputs:
%
%       turb_test     [ny nx nz] or [npoints, 1]  logical array 
%                               The same size as input velocity arrays. True
%                               indicates the the vector is invalid, according
%                               to the constraints set in the turbIntensity
%                               matrix.
%
% Examples:
%       see PIV_3d_vectorcheck.m for example usage in context.
%
% Future Improvements:      none
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
% Created:          29 March 2010
% Last revised:     29 March 2010

%   Copyright (c) 2007-2015  Thomas H. Clark

% CREATE FLUCTUATION VELOCITY ARRAYS
uprime = u - velocity(1);
vprime = v - velocity(2);
wprime = w - velocity(3);


% FACTOR turbIntensity TO GIVE ABSOLUTE VALUES NOT PROPORTIONS
for iCtr = 1:3 %#ok<FORPF>
    if velocity(iCtr) ~= 0
        turbIntensity(:,iCtr) = turbIntensity(:,iCtr)*abs(velocity(iCtr));
    end
end

% DETERMINE MINIMUM FLUCTUATING VELOCITY ACCEPTABLE
% Gives 3-element column vector
minprime = min(turbIntensity,[],2);

% Chuck an error if turbulent intensities are negative or zero
if any(minprime <= 0)
    error('TomoPIVToolbox:NegativeIntensity','Turbulent Intensities cannot be less than or equal to zero')
end

% APPLY TEST CRITERIA
turb_test = uprime > minprime(1);
turb_test = turb_test | (vprime > minprime(2));
turb_test = turb_test | (wprime > minprime(3));













