function [vi] = nearestInterp3(x, y, z, v, xi, yi, zi)
%NEARESTINTERP3 Interps and extrapolates (nearest neighbour) 3D grid data

%   Copyright (c) 2007-2015  Thomas H. Clark

% [dispZ2] = nearestInterp3(guessField.winCtrsX, guessField.winCtrsY, guessField.winCtrsZ, uZ, convLoc(:,1), convLoc(:,2), convLoc(:,3));
            

%   Uses nearest neighbour methods to interpolate and extrapolate data from
%   the known sites x,y,z,v to interpolation locations xi,yi,zi
%
%   x,y,z MUST BE MONOTONICALLY SPACED (to circumvent this, remove the * in
%   the method specificer in the call to interp3 - see code)
%
%   X,Y,Z are plaid uniform grids - NOT produced by meshgrid(), but
%   produced by ndgrid(). xi,yi,zi are column vectors.
%
%   Algorithm:
%       The input grid is padded so that all sites in xi,yi,zi are
%       internal. THen a nearest neighbour3 interpolation is applied.
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

% Debug:
% plot3(x(:),y(:),z(:),'k.');hold on; plot3(xi,yi,zi,'go')

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
    error('MATLAB:nearestInterp3','Input grid is empty, or contains NaNs or Infs')
end

% Check for coplanar input points
if nnz(bsxfun(@eq,x,x(1))) == numel(x)
    error('MATLAB:quickLinInterpExtrap:Coplanar','Coplanar input points in the x direction.')
end
if nnz(bsxfun(@eq,y,y(1))) == numel(y)
    error('MATLAB:quickLinInterpExtrap:Coplanar','Coplanar input points in the y direction.')
end
if nnz(bsxfun(@eq,z,z(1))) == numel(z)
    error('MATLAB:quickLinInterpExtrap:Coplanar','Coplanar input points in the z direction.')
end
    

%% PAD THE INPUT GRID OUT

% Maxima in the input locations and grids
maxX = max(xi(:));
minX = min(xi(:));
maxY = max(yi(:));
minY = min(yi(:));
maxZ = max(zi(:));
minZ = min(zi(:));
minXBound = min(x(:));
maxXBound = max(x(:));
minYBound = min(y(:));
maxYBound = max(y(:));
minZBound = min(z(:));
maxZBound = max(z(:));

% This is a crude technique but useful for this PIV algorithm
ctr = 1;
while (maxX>maxXBound) || (maxY>maxYBound) || (maxZ>maxZBound) || (minX<minXBound) || (minY<minYBound) || (minZ<minZBound)
    
    % pad out the input grid by 1 grid space
    [x y z v] = padOut(x,y,z,v);
    
    % Re-evaluate bounds
    minXBound = min(x(:));
    maxXBound = max(x(:));
    minYBound = min(y(:));
    maxYBound = max(y(:));
    minZBound = min(z(:));
    maxZBound = max(z(:));
    
    % Increment the check ctr and warn for runaways
    ctr = ctr+1;
    if ctr > 10
        warning('MATLAB:TomoPIVToolbox:Runaway','nearestInterp3.m: Extrapolation of the array appears to be running away... xi,yi,zi are too far outside the bounds of hte input grid. Choose a better method!!')
    end
    
end
% hold on; plot3(x(:),y(:),z(:),'bo')

%% INTERPOLATE (all points now within the hull of the data)
vi = interp3(y,x,z,v,yi,xi,zi);


end




% SUBFUNCTION to pad the arrays out by one (nearest neighbour)
function [x,y,z,f1] = padOut(x,y,z,f)

% Pad out in the x,y, and z directions
dx = x(2,1,1)-x(1,1,1);
dy = y(1,2,1)-y(1,1,1);
dz = z(1,1,2)-z(1,1,1);
xvec = x(:,1,1);
yvec = y(1,:,1);
zvec = z(1,1,:);
xvec = [xvec(1)-dx; xvec(:); xvec(end)+dx];
yvec = [yvec(1)-dy; yvec(:); yvec(end)+dy];
zvec = [zvec(1)-dz; zvec(:); zvec(end)+dz];
[x,y,z] = ndgrid(xvec,yvec,zvec);


% Pad the function out using nearest neighbour

f1 = zeros(size(f)+[2 2 2]);

f1(2:end-1,2:end-1,2:end-1) = f;

f1(2:end-1,2:end-1,1)       = f(:,:,1);
f1(2:end-1,2:end-1,end)     = f(:,:,end);

f1(1,2:end-1,2:end-1)       = f(1,:,:);
f1(end,2:end-1,2:end-1)     = f(end,:,:);

f1(2:end-1,1,2:end-1)       = f(:,1,:);
f1(2:end-1,end,2:end-1)     = f(:,end,:);

f1(1,2:end-1,1)             = f(1,:,1);
f1(1,2:end-1,end)           = f(1,:,end);
f1(end,2:end-1,1)           = f(end,:,1);
f1(end,2:end-1,end)         = f(end,:,end);

f1(1,1,2:end-1)             = f(1,1,:);
f1(1,end,2:end-1)           = f(1,end,:);
f1(end,1,2:end-1)           = f(end,1,:);
f1(end,end,2:end-1)         = f(end,end,:);

f1(2:end-1,1,1)             = f(:,1,1);
f1(2:end-1,end,1)           = f(:,end,1);
f1(2:end-1,1,end)           = f(:,1,end);
f1(2:end-1,end,end)         = f(:,end,end);

f1(1,1,1)                   = f(1,1,1);
f1(1,end,1)                 = f(1,end,1);
f1(1,1,end)                 = f(1,1,end);
f1(1,end,end)               = f(1,end,end);
f1(end,1,1)                 = f(end,1,1);
f1(end,end,1)               = f(end,end,1);
f1(end,1,end)               = f(end,1,end);
f1(end,end,end)             = f(end,end,end);

end % end subfunction padOut
