function [] = compare_mart_mfg()

% PREPARE TEST INPUT VARIABLES

%   Copyright (c) 2007-2015  Thomas H. Clark
% Define Cameras
cameras = [1 2 3 4];

% Load the setup variables
setup = load('/home/t_clark/.matlab/tomoPIV/code_tomoreconstruction/mxmart_large/mxmart_large_dev/test_files/test_tomo_setup_file_900800.mat','mask','a','b','c','d','los_factor','vox_X','vox_Y','vox_Z');

% Get a cell containing image file names and images for reconstruction
imfiles{1,1} = '/home/t_clark/.matlab/tomoPIV/code_tomoreconstruction/mxmart_mfg/compare/proc_c1_0000.bmp';
imfiles{2,1} = '/home/t_clark/.matlab/tomoPIV/code_tomoreconstruction/mxmart_mfg/compare/proc_c2_0000.bmp';
imfiles{3,1} = '/home/t_clark/.matlab/tomoPIV/code_tomoreconstruction/mxmart_mfg/compare/proc_c3_0000.bmp';
imfiles{4,1} = '/home/t_clark/.matlab/tomoPIV/code_tomoreconstruction/mxmart_mfg/compare/proc_c4_0000.bmp';
imfiles{1,2} = imread(imfiles{1,1});
imfiles{2,2} = imread(imfiles{2,1});
imfiles{3,2} = imread(imfiles{3,1});
imfiles{4,2} = flipud(fliplr(imread(imfiles{4,1})));

% Normalise the images
for i = 1:1:numel(cameras)
    ind = cameras(i);
    imclass = class(imfiles{ind,2});
    switch lower(imclass)
        case 'uint8'
            imfiles{ind,2} = double(imfiles{ind,2})/255;
        case 'uint16'
            imfiles{ind,2} = double(imfiles{ind,2})/65535;
        otherwise
            error('test_ml_mart_2: image of unrecognised type')
    end
end

% Mask the input images to the pixels we require
pixels = [];
for i = 1:1:numel(cameras)
    ind = cameras(i);
    pixels = [pixels; imfiles{ind,2}(setup.mask{ind})];
end

% Threshold the pixels to increase the sparsity of the problem
pixels(pixels < 0.008) = 0.0;

% Define reconstruction parameters:
weighting = 'circle';
num_mart_iters = 1;
mu = 0.9;
init_intensity = 1;
    


%% CALL THE MART_LARGE MEX FILE WRAPPER
tic
[voxels] = mart_mfg(setup, cameras, pixels, weighting, num_mart_iters, mu, init_intensity);
toc

disp('test_mart_mfg.m: mxmart_mfg test function complete')
% save result_mxmart_mfg_300.mat

end % End main function

