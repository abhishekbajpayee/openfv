function [vi] = quickLinInterpExtrap(x, y, z, v, xi, yi, zi)
%quickLinInterpExtrap Interp from regular grid incl. extrapolation
%   W = quickLinInterpExtrap(X,Y,Z,V,XI,YI,ZI) uses linear interpolation of
%   the regular grid in X,Y,Z and a 'blurred method' extrapolation to place
%   data on grid xi,yi,zi
%
%   x,y,z MUST BE MONOTONICALLY SPACED (to circumvent this, remove the * in
%   the method specificer in the call to interp3 - see code)
%
%   (XI,YI,ZI) and X,Y,Z are plaid uniform grids - NOT produced by
%   meshgrid(), but produced by ndgrid()
%
%   Algorithm:
%       Linear interpolation is used within the convex hull of the data.
%       Outside the convex hull, values are intially set to 0. The volume
%       is blurred, then values which were inside the convex hull are reset
%       to their initially interpolated value. This is repeated several
%       times to diffuse boundary values out from the edges of the hull.
%
%   For scattered input data, try nearestExtrapData3
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
    error('MATLAB:quickLinInterpExtrap','Input grid is empty, or contains NaNs or Infs')
end

% Check for coplanar input points
coplanar = [0 0 0];
if nnz(bsxfun(@eq,x,x(1))) == numel(x)
    error('MATLAB:quickLinInterpExtrap:Coplanar','Coplanar input points in the x direction.')
end
if nnz(bsxfun(@eq,y,y(1))) == numel(y)
    error('MATLAB:quickLinInterpExtrap:Coplanar','Coplanar input points in the y direction.')
end
if nnz(bsxfun(@eq,z,z(1))) == numel(z)
    error('MATLAB:quickLinInterpExtrap:Coplanar','Coplanar input points in the z direction.')
end
    


%% INTERPOLATE WITHIN THE CONVEX HULL

% Leave nans outside
vi = interp3(y,x,z,v,yi,xi,zi,'*linear',NaN);


%% DELAUNAY BASED EXTRAPOLATION

% We don't want the triangulation to include the interior grid - it'll take
% ages. So we cut out, and triangulate only the outermost grid layer and
% the exterior points. Note that we only do it if it actually has interior
% points!
interiorMask = false(size(x));
if (size(x,1)>2) && (size(x,2)>2) && (size(x,3)>2)
    interiorMask(2:end-1,2:end-1,2:end-1) = true;
end

% Mask to store the out-of-hull positions
exteriorMask = isnan(vi);

% Get the points lying on the convex hull
xHull = x(~interiorMask(:));
yHull = y(~interiorMask(:));
zHull = z(~interiorMask(:));
vHull = v(~interiorMask(:));

% Get the exterior points
xExt = xi(exteriorMask(:));
yExt = yi(exteriorMask(:));
zExt = zi(exteriorMask(:));

% Form a triangulation of the points on the convex hull and determine
% nearest neighbours
DT = DelaunayTri(xHull,yHull,zHull);
PI = nearestNeighbor(DT,xExt,yExt,zExt);

% Determine exterior values using nearest neighbour indices
vExt = vHull(PI);

% Parse back into the output
vi(exteriorMask(:)) = vExt;


% %% BLUR TYPE EXTRAPOLATION METHOD
% 
% % Mask to store the out-of-hull positions
% mask = isnan(vi);
% 
% % No point in doing it if they're all OK...
% if any(mask(:))
%     
%     % Initialise to 0
%     vi(mask) = 0;
%     
%     % Store original
%     origvi = vi;
% 
%     % Three blurs, radius 5x5x5, box filter for speed
%     ftype = 'box'
%     filt = [9 9 9];
%     sd  = 9;
%     vi = smooth3(vi,ftype,filt,sd);
%     vi(~mask(:)) = origvi(~mask(:));
%     vi = smooth3(vi,ftype,filt,sd);
%     vi(~mask(:)) = origvi(~mask(:));
%     vi = smooth3(vi,ftype,filt,sd);
%     vi(~mask(:)) = origvi(~mask(:));
%     vi = smooth3(vi,ftype,filt,sd);
%     vi(~mask(:)) = origvi(~mask(:));
%     vi = smooth3(vi,ftype,filt,sd);
%     vi(~mask(:)) = origvi(~mask(:));
%     vi = smooth3(vi,ftype,filt,sd);
%     vi(~mask(:)) = origvi(~mask(:));
%     vi = smooth3(vi,ftype,filt,sd);
%     vi(~mask(:)) = origvi(~mask(:))
% 
% end
