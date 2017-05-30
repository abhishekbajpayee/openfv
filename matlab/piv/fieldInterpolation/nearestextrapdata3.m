function [w] = nearestextrapdata3(x, y, z, v, xi, yi, zi, varargin)
%EXTRAPDATA3 Data gridding and extrapolation of 3D data using hypersurfaces
%   W = EXTRAPDATA3(X,Y,Z,V,XI,YI,ZI) fits a hyper-surface of the form
%   W = F(X,Y,Z) to the data in the (usually) nonuniformly-spaced vectors
%   (X,Y,Z,V).  EXTRAPDATA3 interpolates this hyper-surface at the points
%   specified by (XI,YI,ZI) to produce W. Where points lie outside the convex
%   hull of the input data, their extrapolated values are based on a naturally
%   weighted average of the input points. 
%
%   (XI,YI,ZI) is a plaid uniform grid (as produced by MESHGRID).
%
%   [...] = EXTRAPDATA3(X,Y,Z,V,XI,YI,ZI,METHOD) defines the type of surface fit
%   to the data, where METHOD is one of:
%       'local'     - Tessellation-based linear interpolation (default)
%       'nearest'   - Nearest neighbor interpolation and extrapolation
%
%   defines the type of surface fit to the data. 
%   All the methods are based on a Delaunay triangulation of the data.
%   If METHOD is [], then the default 'linear' method will be used.
%
%   Example:
%
%   Algorithm:
%       The natural weighting function used for extrapolation is 
%       inverse parabolic in distance to the input points: Close to the surface 
%       of the convex hull, extrapolated values tend toward a local mean of the
%       input data. In the limit of infinite distance from the surface, the
%       distance between the extrapolation position and each point in the input
%       dataset tends to the same amount - thus the extrapolated value tends to
%       the mean of the entire dataset. Note that to ensure correct weighting
%       characteristics, the problem is scaled to a unit grid.
%
%	Future Improvements:
%       [1] Input points which are planar in the Z direction are handled, but
%           this handling should be made more robust and extended to x and y
%           planes too.
%       [2] Extension to n-dimensions. For 2 dimensional equivalent, consider
%           'gridfit' by J. D'Errico (available on The Mathworks' File Exchange)
%
%
%   Author Details:          
%
%       T.H. Clark
%
%       The author is available for consultancy (ad-hoc or longer term) in a
%       fields ranging from fluid flow (computational and experimental), MATLAB,
%       computational geometry, financial data modelling and analysis, and 
%       marine (tidal) turbine and propeller design optimisation.
%
%       C.V. and other details can be found at
%           http://cambridge.academia.edu/ThomasClark/
%
%       Fluids Lab
%       Cambridge University Engineering Department
%       2 Trumpington Street
%       Cambridge
%       CB21PZ
%
%       t.clark@cantab.net
%
%   Version Information:
%
%       30 October 2009
%           Created and documented.
%       27 November 2009
%           Addition of rigorous error checks, and handling of coplanar points
%           using extrapdata2.
%
%       Note that EXTRAPDATA3 uses the DelaunayTri and TriScatteredInterp
%       classes introduced in MATLAB R2009a thus is incompatible with earlier
%       versions of MATLAB. Sorry!
%
%   Class support for inputs X,Y,Z,V,XI,YI,ZI: double
%
%   See also TriScatteredInterp, DelaunayTri, GRIDDATA3, QHULL, DELAUNAYN, MESHGRID, GRIDFIT.

%   Copyright (c) 2007-2015  Thomas H. Clark

%% INPUT CHECKS:

% Check for empties, NaNs an Infs:
violation = false;
if isempty(x) || (nnz(isnan(x)) >= 1) || (nnz(isinf(x)) >= 1)
    violation = true;
end
if isempty(y) || (nnz(isnan(y)) >= 1) || (nnz(isinf(y)) >= 1)
    violation = true;
end
if isempty(z) || (nnz(isnan(z)) >= 1) || (nnz(isinf(z)) >= 1)
    violation = true;
end
if violation
    error('MATLAB:extrapdata3','Input data is empty, or contains NaNs or Infs - Invalid data format for DelaunayTri')
end

% Check for coplanar input points
coplanar = [0 0 0];
if nnz(bsxfun(@eq,x,x(1))) == numel(x)
    warning('MATLAB:extrapdata3','Coplanar input points in the x direction. Using extrapdata2.')
    coplanar(1) = 1;
end
if nnz(bsxfun(@eq,y,y(1))) == numel(y)
    warning('MATLAB:extrapdata3','Coplanar input points in the y direction. Using extrapdata2.')
    coplanar(2) = 1;
end
if nnz(bsxfun(@eq,z,z(1))) == numel(z)
    warning('MATLAB:extrapdata3','Coplanar input points in the z direction. Using extrapdata2.')
    coplanar(3) = 1;
end
if nnz(coplanar) >= 2
    % NB This doesn't strictly test for collinearity - it won't pick up the case
    % where points lie on a line which isn't in any of the three planes x=0, y=0
    % or z=0.
    error('MATLAB:extrapdata3','More than one coplanar direction - use a 1D interpolation routine instead')
end

% Check method
method = 'linear';
if nargin > 7
    if ~ischar(varargin{1})
        error('MATLAB:extrapdata3','Method specifier should be a string');
    else
        method = lower(varargin{1});
    end
    if (~strcmp(method,'linear')) && (~strcmp(method,'natural')) && (~strcmp(method,'nearest'))
        error('MATLAB:extrapdata3','Method specifier must be one of ''linear'', ''natural'' or ''nearest''.');
    end
end

%% PASS COPLANAR POINTS TO EXTRAPDATA2

if nnz(coplanar) == 1
        
    % If coplanar in x, then triangulate in y, z etc
    if coplanar(1)
        w = extrapdata2(y,z,v,yi,zi);
    elseif coplanar(2)
        w = extrapdata2(x,z,v,xi,zi);
    else
        w = extrapdata2(x,y,v,xi,yi);
    end
    
    % Re-shape the output to 3D, to match the input arrays
%     w = reshape(w, size(xi));
    
    % Return without doing a 3d triangulation
    return
    
end



%% SCALE DATA FOR GOOD CONDITIONING OF DELAUNAY TRIANGULATION

% NB This produces NaNs if used on coplanar data...

% Scale the input x data between 0 and 1 to condition the weighting properly.
x_min   = min(xi(:));
x_max   = max(xi(:));
x       = x  - x_min;
xi      = xi - x_min;
x       = x  / x_max;
xi      = xi / x_max;

% Same for y values
y_min   = min(yi(:));
y_max   = max(yi(:));
y       = y - y_min;
yi      = yi-y_min;
y       = y/y_max;
yi      = yi/y_max;

% Same for z values
z_min	= min(zi(:));
z_max	= max(zi(:));
z       = z  - z_min;
zi      = zi - z_min;
z       = z  / z_max;
zi      = zi / z_max;

    

%% TRIANGULATE INPUT DATA, AND INTERPOLATE INTERIOR POINTS

% Create a delaunay triangulation OBJECT using the input data:
DT = DelaunayTri(x,y,z);
    
% First, we fit a volume function to the input points. This is easy, using the
% TriScatteredInterp command:
F = TriScatteredInterp(DT, v, method);

% Now determine the interpolated interior and hull-coincident points:
w = F(xi,yi,zi);

% So w is an array of the same size as xi, yi, zi. It contains NaNs where points
% of interpolation lie outside the convex hull.

% Get a mask indicating position of NaN elements, and the number of them...
ext_mask = isnan(w);

% Get column vectors containing x,y,z positions to interpolate to...
ext_xi = xi(ext_mask(:));
ext_yi = yi(ext_mask(:));
ext_zi = zi(ext_mask(:));

% Get the nearest neighbour information
PI = nearestNeighbor(DT,ext_xi, ext_yi, ext_zi);

% Use nearest neighbour information to extrapolate and replace NaNs
wFilled = w;
wFilled(ext_mask(:)) = v(PI);

% % Smooth the crap out of it - ONLY IF 3D
% wFilled = smooth3(reshape(wFilled, size(xi)),'box', 3);
% 
% % Parse back in the good values
% wFilled(~ext_mask(:)) = w(~ext_mask(:));

% output
w = wFilled;
