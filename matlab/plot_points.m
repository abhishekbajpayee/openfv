clear all;

[cam_id, pts_id, u, v] = textread('../obervations.txt');

cam = 0;
pt_id = 0;
ppi = 30;
img = 2;

ind = find(cam_id==cam);
pts_f = pts_id(ind);
u_f = u(ind);
v_f = v(ind);

filename = ['../../experiment/calibration_rect/cam' num2str(cam+1) '/' num2str(img*5,'%.2d') '.jpg']
I = imread(filename);
imshow(I);
hold on;
scatter(u_f((img*ppi)+1:(img*ppi)+ppi)+1, v_f((img*ppi)+1:(img*ppi)+ppi)+1,'r+');

% ind = find(pts_f==pt_id);
% pts_f = pts_f(ind);
% u_f = u_f(ind);
% v_f = v_f(ind);

% figure;
% plot(u_f,v_f);

[pts] = textread('../world_points.txt');

x = [];
y = [];
z = [];
for i=1:ppi*9
    x = [x pts( ((i-1)*3)+1 )];
    y = [y pts( ((i-1)*3)+2 )];
    z = [z pts( ((i-1)*3)+3 )];
end
figure;
scatter3(x, y, z);