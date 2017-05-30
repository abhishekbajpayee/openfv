function [] = test_whitInterp3

% Testing the sinc interpolation function

%   Copyright (c) 2007-2015  Thomas H. Clark
% First, create a matrix of zeros with ones in strategic locations
% field = zeros(32,32,32);
% field(3,3,3) = 1;
% field(30,3,3) = 1;
% field(30,3,30) = 1;

% Determine the window corner locations for the first test.
% This requires knowledge of the ORDER in which corner indices are
% stored in the wCorner vectors. Order is as follows:
%          0       low row, low col, low page
%          1       high row, low col, low page
%          2       high row, high col, low page
%          3       low row, high col, low page
%          4       low row, low col, high page
%          5       high row, low col, high page
%          6       high row, high col, high page
%          7       low row, high col, high page
%          8       repeat for next window in same order

% First, we attempt retriving the inner array. NB need 2 voxel 'edge cut' for
% the interpolation algorithm so we can't retrieve the whole array
% c0 = [3  3  3]+[0.5 0.5 0.5];
% c1 = [3  30 3]+[0.5 -0.5 0.5];
% c2 = [30 30 3]+[-0.5 -0.5 0.5];
% c3 = [30  3 3]+[-0.5 0.5 0.5];
% c4 = [3  3  30]+[0.5 0.5 -0.5];
% c5 = [3  30 30]+[0.5 -0.5 -0.5];
% c6 = [30 30 30]+[-0.5 -0.5 -0.5];
% c7 = [30  3 30]+[-0.5 0.5 -0.5];
% 
% wSize = [28 28 28];

% Call sincInterp3 function
% [window] = sincInterp3(field, c0, c1, c2, c3, c4, c5, c6, c7, wSize);
% 
% disp(['Window [1  1  1 ] = ' num2str(window(1,1,1))])
% disp(['Window [28 1  1 ] = ' num2str(window(28,1,1))])
% disp(['Window [28 28 1 ] = ' num2str(window(28,28,1))])
% disp(['Window [1  28 1 ] = ' num2str(window(1,28,1))])
% disp(['Window [1  1  28] = ' num2str(window(1,1,28))])
% disp(['Window [28 1  28] = ' num2str(window(28,1,28))])
% disp(['Window [28 28 28] = ' num2str(window(28,28,28))])
% disp(['Window [1  28 28] = ' num2str(window(1,28,28))])

% Load more challenging data and examine singularity regions
% c0 = [8  8  8]              +[0.0 0.0 0.0];
% c1 = [8  10 8]             +[0.0 -0.0 0.0];
% c2 = [10 10 8]         +[-0.0 -0.0 0.0];
% c3 = [10  8 8]        +[-0.0 0.0 0.0];
% c4 = [8  8  10]           +[0.0 0.0 -0.0];
% c5 = [8  10 10]         +[0.0 -0.0 -0.0];
% c6 = [10 10 10]      +[-0.0 -0.0 -0.0];
% c7 = [10  8 10]           +[-0.0 0.0 -0.0];
%


sincTestArray = positionGaussian(zeros(48,48,48), [9 9 9], 1);
sincTestArray = sincTestArray/(max(sincTestArray(:)));
x = linspace(1,48,size(sincTestArray,1));
[X Y Z] = ndgrid(x,x,x);

% wSize of 41 chosen so that points pass through singularities
wSize = [43 43 43];
tic
[window windowOrig] = whitInterp3(sincTestArray, wSize);
toc
x2 = linspace(7.95,10.05,wSize(1));
[X2 Y2 Z2] = ndgrid(x2,x2,x2);


raiseFigure('whittaker interpolation test');
clf; hold on
% ph1 = patch(isosurface(X, Y, Z, sincTestArray,0.6));
ph2 = patch(isosurface(X2,Y2,Z2,window,      0.64));
% ph3 = patch(isosurface(X2,Y2,Z2,windowOrig,      0.64));

set(ph2,'faceColor','b','faceAlpha',0.5,'edgeAlpha',0)
% set(ph3,'faceColor','g','faceAlpha',0.3,'edgeAlpha',0)

camlight right
lighting phong

axis equal
xlim([8 10])
ylim([8 10])
zlim([8 10])




% SUBROUTINE TO ADD A GAUSSIAN PEAK TO THE INTENSITY ARRAY
function intensity = positionGaussian(intensity, position, sigma)

    % Distance from the gaussian position
    [coord1 coord2 coord3] = ndgrid(1:size(intensity,1),1:size(intensity,2), 1:size(intensity,3));
    coord1 = coord1 - position(1);
    coord2 = coord2 - position(2);
    coord3 = coord3 - position(3);
    radius = sqrt(coord1.^2 + coord2.^2 + coord3.^2);
    
    % Gaussian Function:
    intensity = intensity + exp(-0.5 * (radius./sigma).^2) ./ (sqrt(2*pi) .* sigma);
    
