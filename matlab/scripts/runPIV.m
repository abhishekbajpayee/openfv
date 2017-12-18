% Function to read data in specified format from path variable and
% then run PIV based on other specified settings

function velocityField = runPIV(data_path, frames, pivOpts, f, dt)

for j=1:2

    tmp_path = [data_path frames{j} '/'];
    imn1 = dir([tmp_path '*.tif']);

    if j==1
        img = imread([tmp_path imn1(1).name]);
        stacks = zeros(size(img, 1), size(img, 2), size(imn1, 1), ...
                       2);
    end
    
    for i=1:size(imn1, 1)
        stacks(:,:,i,j+1) = imread([tmp_path imn1(i).name]);
    end

end

[vx, vy, vz, nf] = size(stacks);
voxels.vox_X = linspace(-0.5*vx/f, 0.5*vx/f, vx);
voxels.vox_Y = linspace(-0.5*vy/f, 0.5*vy/f, vy);
voxels.vox_Z = linspace(-0.5*vz/f, 0.5*vz/f, vz);

% Get the individual field variables and lose temporary variables
% Note: not scaling the stack because already in a [0,255] range
fieldA = double(squeeze(stacks(:,:,:,1)));
fieldB = double(squeeze(stacks(:,:,:,2)));
clearvars i j stacks img tmp_path imn1 path

% Calling PIV
velocityField = PIV_3d(fieldA, fieldB, voxels, dt, pivOpts);
    