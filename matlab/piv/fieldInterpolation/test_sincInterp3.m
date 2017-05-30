

% Testing the sinc interpolation function

%   Copyright (c) 2007-2015  Thomas H. Clark
% First, create a matrix of zeros with ones in strategic locations
field = zeros(32,32,32);
field(3,3,3) = 1;
field(30,3,3) = 1;
field(30,3,30) = 1;

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
c0 = [3  3  3]+[0.5 0.5 0.5];
c1 = [3  30 3]+[0.5 -0.5 0.5];
c2 = [30 30 3]+[-0.5 -0.5 0.5];
c3 = [30  3 3]+[-0.5 0.5 0.5];
c4 = [3  3  30]+[0.5 0.5 -0.5];
c5 = [3  30 30]+[0.5 -0.5 -0.5];
c6 = [30 30 30]+[-0.5 -0.5 -0.5];
c7 = [30  3 30]+[-0.5 0.5 -0.5];

wSize = [28 28 28];

% Call sincInterp3 function
[window] = sincInterp3(field, c0, c1, c2, c3, c4, c5, c6, c7, wSize);

disp(['Window [1  1  1 ] = ' num2str(window(1,1,1))])
disp(['Window [28 1  1 ] = ' num2str(window(28,1,1))])
disp(['Window [28 28 1 ] = ' num2str(window(28,28,1))])
disp(['Window [1  28 1 ] = ' num2str(window(1,28,1))])
disp(['Window [1  1  28] = ' num2str(window(1,1,28))])
disp(['Window [28 1  28] = ' num2str(window(28,1,28))])
disp(['Window [28 28 28] = ' num2str(window(28,28,28))])
disp(['Window [1  28 28] = ' num2str(window(1,28,28))])

% Load more challenging data
% load sincInterpTestData.mat
% [window2] = sincInterp3(sincTestArray, c0, c1, c2, c3, c4, c5, c6, c7, wSize)
% nnz(window2)
% fv = isosurface(X,Y,Z,V,isovalue)



