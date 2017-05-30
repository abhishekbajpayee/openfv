function [w] = extrapdata2(x, y, v, xi, yi)
%EXTRAPDATA2 Data gridding and extrapolation of 2D data using hypersurfaces
%   W = EXTRAPDATA2(X,Y,2V,XI,YI2) fits a surface of the form W = F(X,Y) to the
%   data in the (usually) nonuniformly-spaced vectors (X,Y,V).  EXTRAPDATA2
%   interpolates this hyper-surface at the points specified by (XI,YI) to
%   produce W. Where points lie outside the convex hull of the input data, their
%   extrapolated values are based on a naturally weighted average of the input
%   points. 
%
%   (XI,YI) is a plaid uniform grid (as produced by MESHGRID).
%
%   [...] = EXTRAPDATA2(X,Y,V,XI,YI,METHOD) defines the type of surface fit
%   to the data, where METHOD is one of:
%       'local'     - Tessellation-based linear interpolation (default)
%       'nearest'   - Nearest neighbor interpolation
%
%   Both methods are based on a Delaunay triangulation of the data.
%   If METHOD is [], then the default 'linear' method will be used.
%
%   Alternatives:
%       Use gridfit by J. D'Errico, available on www.mathworks.com file
%       exchange. Gridfit has a far wider variety of options than extrapdata2.
%       The purpose of extrapdata2 is to provide an analagous method for the
%       extrapdata3 command, which works for Colinear points.
%
%
%   Extrapolation Algorithm:
%       The natural weighting function used for extrapolation is 
%       inverse parabolic in distance to the input points: Close to the surface 
%       of the convex hull, extrapolated values tend toward a local mean of the
%       input data. In the limit of infinite distance from the surface, the
%       distance between the extrapolation position and each point in the input
%       dataset tends to the same amount - thus the extrapolated value tends to
%       the mean of the entire dataset. Note that to ensure correct weighting
%       characteristics, the problem is scaled to a unit grid.
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
%       27 November 2009
%           Created.
%
%       Note that EXTRAPDATA2 uses the DelaunayTri and TriScatteredInterp
%       classes introduced in MATLAB R2009a thus is incompatible with earlier
%       versions of MATLAB. Sorry!
%
%      
%
%   Class support for inputs X,Y,V,XI,YI: double
%
%   See also TriScatteredInterp, DelaunayTri, GRIDDATA, QHULL, DELAUNAYN, MESHGRID, GRIDFIT.

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
if violation
    error('MATLAB:extrapdata2','Input data is empty, or contains NaNs or Infs - Invalid data format for DelaunayTri')
end

% Check for colinear input points
if nnz(bsxfun(@eq,x,x(1))) == numel(x)
    error('MATLAB:extrapdata2','Colinear data in the x direction - use a 1D interpolation routine instead')
end
if nnz(bsxfun(@eq,y,y(1))) == numel(y)
    error('MATLAB:extrapdata2','Colinear data in the y direction - use a 1D interpolation routine instead')
end




%% SCALE DATA FOR GOOD CONDITIONING OF DELAUNAY TRIANGULATION

% NB This produces NaNs if used on colinear data...

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
    

%% TRIANGULATE INPUT DATA, AND INTERPOLATE INTERIOR POINTS

% Create a delaunay triangulation OBJECT using the input data:
DT = DelaunayTri(x,y);
    
% First, we fit a volume function to the input points. This is easy, using the
% TriScatteredInterp command:
F = TriScatteredInterp(DT, v,'linear');

% Now determine the interpolated interior and hull-coincident points:
w = F(xi,yi);

% So w is an array of the same size as xi, yi, zi. It contains NaNs where points
% of interpolation lie outside the convex hull.

% For each position outside the convex hull, we wish to perform an
% extrapolation. For each point in the input dataset, we require the distance^2
% between it and the extrapolation position.

% Get a mask indicating position of NaN elements, and the number of them...
ext_mask = isnan(w);
n_ext = nnz(ext_mask(:));

% Get column vectors containing x,y positions to interpolate to...
ext_xi = xi(ext_mask(:));
ext_yi = yi(ext_mask(:));

% For extrapolation, the already interpolated data is used as a basis. If the
% input data is used, then there is a discontinuity between the extrapolated
% data and the interpolated data on the boundary of the convex hull.
% NB transpose into row vector form for use with bsxfun, below.
int_xi = xi(~ext_mask(:))';
int_yi = yi(~ext_mask(:))';
int_wi = w(~ext_mask(:))';


% Use binary singleton expansion to create [n_ext x n_int] matrices:
x_dist_sqd = bsxfun(@minus,ext_xi,int_xi).^2;
y_dist_sqd = bsxfun(@minus,ext_yi,int_yi).^2;


dx = yi(2)-yi(1);
dist = (sqrt(x_dist_sqd + y_dist_sqd))./dx;

weight = 1./((2/dx).^dist);
% weight = 1-(1./ (weight./max(weight(:))));

% The distance^2 is used as the weighting function...
% weight = 0.5*(x_dist_sqd + y_dist_sqd); % varies between 0 and 1

% Close together elements should have a higher weighting. Thus we need to take
% the 1/x of the weights
% weight = 1./weight;


% We'll need to normalise back at some point. Sum elements along rows:
weight_sum = sum(weight,2);

% Perform the weighted averaging
ext_wi = sum(weight.*repmat(int_wi,[n_ext 1]),2)./weight_sum;

% Index back into the output array
w(ext_mask) = ext_wi;









