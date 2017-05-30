

% Create plaid grids
[xi yi] = meshgrid(-10:0.1:5,-10:0.1:5);
zi = zeros(size(xi));

%   Copyright (c) 2007-2015  Thomas H. Clark
% Create a 'true' hypersurface
vi = peaks(xi,yi);
vi = vi./max(vi(:));
meanval = mean(vi(:))

% Create distorted / corrupted data (missing elements, one corner chopped off)
null_data_mask = (rand(size(xi)) > 0.9) | ((xi >= 0.2) & (yi > 0.6));
x = xi(:);
y = yi(:);
z = zi(:);
v = vi(:);
x = x(~null_data_mask(:));
y = y(~null_data_mask(:));
z = z(~null_data_mask(:));
v = v(~null_data_mask(:));

% Use extrapdata2 to interpolate and extrapolate
vi_reconstructed = extrapdata3(x,y,z,v,xi,yi,zi);

% Plot the results
fh = figure(2056);
clf
set(fh,'Name','Test of extrapdata2')
set(fh,'NumberTitle','off')
orig_h = surf(xi,yi,vi);
hold on
set(orig_h,'FaceColor',[0 0 1])
corrupted_h = plot3(x,y,v,'go');
recon_h = surf(xi,yi,vi_reconstructed);
set(recon_h,'FaceColor',[0 1 0])
alpha(0.5)
camlight right
lighting phong
