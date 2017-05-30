function [disp_test] = PIV_3d_maxdisplacement(ux, uy, uz, max_ux, max_uy, max_uz)
%PIV_3D_MAXDISPLACEMENT Vector field validation using max particle displacement
%   Simple approach: Where velocities (displacements) exceed the limiting amount
%   in x, y, or z component, they are labelled as spurious. NB the absolute
%   value of velocity is used (i.e. algorithm is independant of velocity sign).
%   
%
% Syntax:  
%       [disp_test] = PIV_3d_maxdisplacement(ux, uy, uz, max_ux, max_uy, max_uz)
%
% Inputs:
%       ux      [ny, nx, nz]    3D array containing x-components of velocity
%                               field.
%
%       uy      [ny, nx, nz]    3D array containing y-components of velocity
%                               field.   
%
%       uy      [ny, nx, nz]    3D array containing z-components of velocity
%                               field.
%
%       max_ux, max_uy, max_uz  [1 x 1] 
%                               Maximum velocities (or displacements) allowable
%                               in each direction.
%
% Outputs:
%
%       disp_test   [ny nx nz]  logical array 
%                               The same size as input velocity arrays. True
%                               indicates the the vector is invalid, according
%                               to the above criterion.
%
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
% Created:          30 October 2009
% Last revised:     30 October 2009

%   Copyright (c) 2007-2015  Thomas H. Clark
% Take the absolute value of velocity
ux = abs(ux);
uy = abs(uy);
uz = abs(uz);


% logical test:
disp_test = (ux > max_ux) | (uy > max_uy) | (uz > max_uz);















