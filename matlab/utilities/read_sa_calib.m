% Function to read in an SA calibration (or OpenFV compatible) file
% and return the contained

function [setupInfo] = read_sa_calib(filename)

file = fopen(filename, 'r');

time_stamp = fgetl(file);

data = fscanf(file, '%f');
setupInfo.imx = data(2);
setupInfo.imy = data(3);
setupInfo.scale = data(4);
num_cams = data(5);
setupInfo.num_cams = data(5);

setupInfo.cam_names = cell(1, num_cams);
setupInfo.Pmats = cell(1, num_cams);
setupInfo.cam_locs = cell(1, num_cams);
for i=1:num_cams
    setupInfo.cam_names{i} = fgetl(file);
    setupInfo.Pmats{i} = fscanf(file, '%f', [4, 3]);
    setupInfo.Pmats{i} = setupInfo.Pmats{i}';
    setupInfo.cam_locs{i} = fscanf(file, '%f');
end

ref_data = setupInfo.cam_locs{num_cams}(4:end);
setupInfo.cam_locs{num_cams} = setupInfo.cam_locs{num_cams}(1:3);

setupInfo.ref.on = false;
if ref_data(1) == 1
    setupInfo.ref.on = true;
    setupInfo.ref.geom = ref_data(2:end);
end

end