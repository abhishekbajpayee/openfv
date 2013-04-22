clear all;

cam_params = textread('../cameras.txt');

x = -25;
y = -20;
z = -20;

rvec = cam_params(1:3);
R = vrrotvec2mat(rvec);
