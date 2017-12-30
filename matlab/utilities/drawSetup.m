% This draws the scene along with cameras based on the calibration
% that has been setup. Mostly only used at debugging stage.

function drawSetup(setup, setupInfo)

%% World coordinates scene

% TODO: check for back projection error
% TODO: somehow plot volume of intersection?

figure;
hold on;
daspect([1 1 1]);
s = 150;
axis([-s s -s s -s s])

cam_locs = setupInfo.cam_locs;
imx = setupInfo.imx;
imy = setupInfo.imy;
ref = setupInfo.ref;

ncams = size(cam_locs, 2);
ids = [1, imx, imx*imy-imx+1, imx*imy];
for cam = 1:ncams

    pts = setup.a{cam}';
    dvec = setup.b{cam}';

    if ref.on
        [projA, projB, err] = img_refrac(cam_locs{cam}, pts, ...
                                         ref.geom(1), ref.geom(2), ...
                                         ref.geom(3:end));
    end

    for i = 1:size(ids, 2)

        if ref.on
            plot3([cam_locs{cam}(1), projA(1,ids(i)), projB(1,ids(i)), pts(1,ids(i))], ...
                  [cam_locs{cam}(2), projA(2,ids(i)), projB(2,ids(i)), pts(2,ids(i))], ...
                  [cam_locs{cam}(3), projA(3,ids(i)), projB(3,ids(i)), pts(3,ids(i))]);
        else
            plot3([cam_locs{cam}(1), pts(1,ids(i))], ...
                  [cam_locs{cam}(2), pts(2,ids(i))], ...
                  [cam_locs{cam}(3), pts(3,ids(i))]);
        end
        
        a = 50;
        arrow3(pts(:,ids(i))', pts(:,ids(i))' + dvec(:,ids(i))'*a);

    end
    
end

face = [setup.vox_X(1),   setup.vox_Y(1),   setup.vox_Z(1); ...
        setup.vox_X(end), setup.vox_Y(1),   setup.vox_Z(1); ...
        setup.vox_X(end), setup.vox_Y(end), setup.vox_Z(1); ...
        setup.vox_X(1),   setup.vox_Y(end), setup.vox_Z(1)];

back = [setup.vox_X(1),   setup.vox_Y(1),   setup.vox_Z(end); ...
        setup.vox_X(end), setup.vox_Y(1),   setup.vox_Z(end); ...
        setup.vox_X(end), setup.vox_Y(end), setup.vox_Z(end); ...
        setup.vox_X(1),   setup.vox_Y(end), setup.vox_Z(end)];

fill3(face(:,1),face(:,2),face(:,3),0.5);
alpha(0.5);
fill3(back(:,1),back(:,2),back(:,3),0.2);
alpha(0.5);

if ref.on
    
    f = 5;
    wall1 = [f*setup.vox_X(1),   f*setup.vox_Y(1),   ref.geom(1); ...
             f*setup.vox_X(end), f*setup.vox_Y(1),   ref.geom(1); ...
             f*setup.vox_X(end), f*setup.vox_Y(end), ref.geom(1); ...
             f*setup.vox_X(1),   f*setup.vox_Y(end), ref.geom(1)];

    wall2 = [f*setup.vox_X(1),   f*setup.vox_Y(1),   ref.geom(1)+ref.geom(2); ...
             f*setup.vox_X(end), f*setup.vox_Y(1),   ref.geom(1)+ref.geom(2); ...
             f*setup.vox_X(end), f*setup.vox_Y(end), ref.geom(1)+ref.geom(2); ...
             f*setup.vox_X(1),   f*setup.vox_Y(end), ref.geom(1)+ref.geom(2)];


    fill3(wall1(:,1),wall1(:,2),wall1(:,3),0.1);
    alpha(0.2);
    fill3(wall2(:,1),wall2(:,2),wall2(:,3),0.15);
    alpha(0.2);
    
end

xlabel('x');
ylabel('y');
zlabel('z');

%% Scaled scene

% figure;
% hold on;
% daspect([1 1 1]);
% s = 400;
% axis([-200 800 -200 800 -200 800])

% scale = setup.c{1}(1)/setup.a{1}(1);

% face = face*scale;
% back = back*scale;

% n = size(setup.mask, 1);
% j=1;
% for i=1:n
    
%     l = size(setup.vox_Z, 2);
%     arrow3(setup.c{i}(j,:), setup.d{i}(j,:) + l*setup.d{i}(j,:));
    
% end

% fill3(face(:,1),face(:,2),face(:,3),0.2);
% alpha(0.5);
% fill3(back(:,1),back(:,2),back(:,3),0.5);
% alpha(0.5);

% xlabel('x');
% ylabel('y');
% zlabel('z');