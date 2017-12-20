function [] = checkview_mfg()


%% LOAD MFG RESULT

%   Copyright (c) 2007-2015  Thomas H. Clark
load result_mfg setup voxels


% figure()
% hister = voxels(voxels>2*eps('single'));
% hist(hister,2000)

%% PLOT VOLUME BOUNDING BOX
figure()
hold on
NVOX_X = numel(setup.vox_X);
NVOX_Y = numel(setup.vox_Y);
NVOX_Z = numel(setup.vox_Z);
volmat_x = [1 NVOX_X NVOX_X 1 1 1 NVOX_X NVOX_X NVOX_X NVOX_X NVOX_X NVOX_X 1 1 1 1];
volmat_y = [1 1 NVOX_Y NVOX_Y 1 1 1 1 1 NVOX_Y NVOX_Y NVOX_Y NVOX_Y NVOX_Y NVOX_Y 1];
volmat_z = [1 1 1 1 1 NVOX_Z NVOX_Z 1 NVOX_Z NVOX_Z 1 NVOX_Z NVOX_Z 1 NVOX_Z NVOX_Z];
plot3(volmat_x,volmat_y,volmat_z,'m-')


%% PLOT MFG RESULT

% [globX,globY,globZ] = meshgrid(single(1:numel(setup.vox_X)),single(1:numel(setup.vox_Y)),single(1:numel(setup.vox_Z)));
[globX,globY,globZ] = meshgrid(single(64:128),single(64:128),single(64:128));
%iso = mean(voxels)/10;
voxels = reshape(voxels, numel(setup.vox_Y), numel(setup.vox_X), numel(setup.vox_Z));

%figure()
fv = isosurface(globX,globY,globZ,voxels(64:128,64:128,64:128),0.00001);
patch(fv)

axis equal
sparsity = sum(sum(sum(voxels>2*eps('single'))))/(NVOX_X*NVOX_Y*NVOX_Z);
disp(['Sparsity factor for mfg is: ' num2str(sparsity)])

clearvars fv voxels



%% LOAD AND PLOT LARGE RESULT
load result_large voxels

voxels = reshape(voxels, numel(setup.vox_Y), numel(setup.vox_X), numel(setup.vox_Z));
fv = isosurface(globX,globY,globZ,voxels(64:128,64:128,64:128),0.0001);
patch(fv)

nnz_voxels = sum(sum(sum(voxels>2*eps('single'))));
sparsity = nnz_voxels/(NVOX_X*NVOX_Y*NVOX_Z);
disp(['Number of nonzero voxels (using mxmart_large) is: ' num2str(nnz_voxels)])
disp(['Sparsity factor for large is: ' num2str(sparsity)])



