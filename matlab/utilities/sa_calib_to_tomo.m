% Function to convert SA calibration data to a struct that contains
% everything needed to run a Tomo reconstruction

function [setup] = sa_calib_to_tomo(setupInfo, voxels, bounds)

Pmats = setupInfo.Pmats;
cam_locs = setupInfo.cam_locs;
imx = setupInfo.imx;
imy = setupInfo.imy;
scale = setupInfo.scale;
num_cams = setupInfo.num_cams;
ref = setupInfo.ref;
    
setup.vox_X = linspace(bounds(1,1), bounds(1,2), voxels(1));
setup.vox_Y = linspace(bounds(2,1), bounds(2,2), voxels(2));
setup.vox_Z = linspace(bounds(3,1), bounds(3,2), voxels(3)); 

% hold on;
setup.mask = cell(num_cams, 1);
setup.a = cell(num_cams, 1);
setup.b = cell(num_cams, 1);
setup.c = cell(num_cams, 1);
setup.d = cell(num_cams, 1);
setup.los_factor = cell(num_cams, 1);
setup.pvr = cell(num_cams, 1);

[x, y] = meshgrid(0:imx-1,0:imy-1);
points = zeros(3,imy,imx);
points(1,:,:) = x;
points(2,:,:) = y;
points(3,:,:) = ones(imy, imx);

for cam = 1:num_cams;

    setup.mask{cam} = logical(ones(imy*imx, 1));
    
    % NOTE: wonder if this needs to be accurate
    setup.pvr{cam} = setup.mask{cam};
    
    if ref.on
        
        [setup.a{cam}, in_pts] = back_project_ref_H(points, bounds(3,1), ...
                                                    cam_locs{cam}, Pmats{cam}, ...
                                                    ref);
        
        setup.b{cam} = setup.a{cam} - in_pts;
        setup.b{cam} = normc(setup.b{cam});
        
        setup.a{cam} = setup.a{cam}';
        setup.b{cam} = setup.b{cam}';
                
    else
        setup.a{cam} = back_project(points, Pmats{cam}, bounds(3,1));
        setup.b{cam} = calc_b(setup.a{cam}, cam_locs{cam})';
        setup.a{cam} = setup.a{cam}';
    end
    
    setup.c{cam} = setup.a{cam};
    setup.c{cam}(:,1) = (setup.c{cam}(:,1) - setup.vox_X(1))*scale;
    % NOTE: check why this is inverted in original code
    setup.c{cam}(:,2) = (setup.c{cam}(:,2) - setup.vox_Y(1))*scale;
    setup.c{cam}(:,3) = (setup.c{cam}(:,3) - setup.vox_Z(1))*scale;

    setup.d{cam} = setup.b{cam};
    % setup.d{cam}(:,1) = -setup.d{cam}(:,1);
    lam = 1./setup.d{cam}(:,3);
    setup.d{cam} = repmat(lam,1,3).*setup.d{cam};
    
    los_fac_x = cos(atan( setup.d{cam}(:,1)./setup.d{cam}(:,3) ))';
    los_fac_y = cos(atan( setup.d{cam}(:,2)./setup.d{cam}(:,3) ))';
    setup.los_factor{cam} = [los_fac_x;los_fac_y];

end

end
