function [ux uy uz nan_mask] = PIV_3d_gapfill(mesh_x, mesh_y, mesh_z, ux, uy, uz, varargin)
%PIV_3D_GAPFILL Uses a hypersurface fit to replace NaNs in a velocity field
%
%   Velocity fields returned from PIV_3d_vectorcheck may still contain NaN
%   elements. These represent gaps in the flow which arise usually due to poorly
%   seeded areas, regions where there is no light, or reflections (from walls
%   etc).
%
%   When plotting quiver diagrams of the velocity field, this is not a problem:
%   vectors containing NaN components are simply not plotted. However, where
%   vector operators must be applied, or some whole-field technique (such as
%   smoothing or interpolation to a finer grid) is used then we cannot tolerate
%   a non-monotonic vector field.
%
%   It is usually suggested that missing values are replaced using either:
%       - linear interpolation
%       - iterative local mean calculation (similar to linear interpolation)
%       - some flow-dependent criterion (e.g constraint that divergence = 0).
%
%   Here, we take advantage of the efficiency of a delaunay triangulation in
%   order to produce linear interpolation.
%
%   Limitation: This does not extrapolate beyond the convex hull of the valid
%   data. So further treatment is suggested for elements outside the convex
%   hull.
%
% Syntax:
%
%       [ux uy uz nan_mask] = PIV_3d_gapfill(mesh_x, mesh_y, mesh_z, ux, uy, uz, method)
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
%   [1]     Addition of an iterative local mean (or local median) approach where
%           extrapolating data to the edges of the volume
%
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
% Revision History:     30 October 2009     Created
%                       16 March 2011       Altered reshaping of inputs to
%                                           identical operation but with clearer
%                                           syntax

%   Copyright (c) 2007-2015  Thomas H. Clark
% Check for invalid input data set (nb we get segmentation violation if we put
% arrays of NaNs into qhull)
if nnz(~isnan(ux)) <= 3
    % Then there are only three non-NaN points: the convex hull used in the
    % gap-filling algorithm becomes coplanar and the algorithm is
    % ill-conditioned. Thus, we simply replace null velocities with zeroes,
    % and smooth the fields, hoping that the next pass will be more successful.
    nan_mask = isnan(ux);
    ux(nan_mask) = 0;
    uy(nan_mask) = 0;
    uz(nan_mask) = 0;
    ux = smooth3(ux,'gaussian');
    uy = smooth3(uy,'gaussian');
    uz = smooth3(uz,'gaussian');
    warning('TomoPIVToolbox:IllConditionedProblem','There are less than 4 non-NaN vectors in this field. Results are likely to be incorrect.')
    return
end
    
debug = false;
if debug
    
    raiseFigure('PIV_3d_gapfill debugging plot');
    clf

    quiver3(mesh_x, mesh_y, mesh_z, ux, uy, uz,'b')
    hold on
    drawnow
end

                    
% Reshape input velocity fields to column vectors:
col_x = mesh_x(:);
col_y = mesh_y(:);
col_z = mesh_z(:);
ux = ux(:);
uy = uy(:);
uz = uz(:);

% Determine the mask containing NaN elements
nan_mask = isnan(ux) | isnan(uy) | isnan(uz) | isinf(ux) | isinf(uy) | isinf(uz) ;

% Eliminate the elements of the column vectors which have NaN or Infs
col_x(nan_mask) = [];
col_y(nan_mask) = [];
col_z(nan_mask) = [];
ux(nan_mask) = [];
uy(nan_mask) = [];
uz(nan_mask) = [];

% Sort the method to use dependant on the number of arguments parsed
if nargin == 7
    % nb error checking within the extrapdata3 code...
    method = varargin{1};
else
    method = 'linear';
end

% Perform the interior point gap filling, using the extrapdata function (similar
% to griddata, but including locally weighted extrapolation)
[ux] = extrapdata3(col_x, col_y, col_z, ux, mesh_x, mesh_y, mesh_z, method);
[uy] = extrapdata3(col_x, col_y, col_z, uy, mesh_x, mesh_y, mesh_z, method);
[uz] = extrapdata3(col_x, col_y, col_z, uz, mesh_x, mesh_y, mesh_z, method);

% Debug plot
if debug
    
    quiver3(mesh_x, mesh_y, mesh_z, ux, uy, uz,'g')
    legend({'Input Vector Field';'Gap-filled vector field'})
end
