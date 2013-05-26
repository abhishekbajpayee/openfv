%   Written by Jesse Belden 
%   jbelden@mit.edu
%   March 31, 2009
%
% This program simulates the aquisition of images of a seeded particle volume using a camera array.
% The camera array can be arbitrarily configured.  Particles are randomly distributed through a 
% user-defined volume.  Geometric optics are used to model the image aquisition, and the entire
% field is considered to be in focus.  The lenses are modeled as pinholes; i.e., there are no lens
% aberrations.

function [XP2] = synthetic_3d_vort_ring_generator_YZ_plane(XP,Ro,L,ZR,C)


% R  = 0:0.01:10;  %Radial distance of point from center of vortex ring
% d  = (8*R*C/L).*exp(-R/L);

%TEST PARTICLES TO MAKE SURE CODE IS WORKING
% % % ang = 0;
% % % XP1 = zeros(1,11) + Ro*sin(ang);
% % % XP2 = zeros(1,11) + Ro*cos(ang);
% % % XP3 = 0.1:1:10.1;
% % % 
% % % % %Change angle
% % % % for j = 1:7
% % % %     ang = ang + pi/4;
% % % %     XP1 = [XP1 zeros(1,11) + Ro*sin(ang)];
% % % %     XP2 = [XP2 zeros(1,11) + Ro*cos(ang)];
% % % %     XP3 = [XP3 0.1:1:10.1];
% % % % end
% % % 
% % % %Change distance
% % % A = Ro;
% % % for j = 1:11
% % %     A = A-2;
% % %     XP1 = [XP1 zeros(1,11) + A*sin(ang)];
% % %     XP2 = [XP2 zeros(1,11) + A*cos(ang)];
% % %     XP3 = [XP3 0.1:1:10.1];
% % % end
% % % 
% % % [XP1,XP2] = meshgrid(-10:4:10,-10:4:10);
% % % [m,n] = size(XP1);
% % % XP1 = reshape(XP1,[m*n,1]);XP1 = XP1';
% % % XP2 = reshape(XP2,[m*n,1]);XP2 = XP2';
% % % XP3 = zeros(size(XP1)) + ZR;XP3 = XP3';
% % % 
% % % XP(1,:) = XP1;
% % % XP(2,:) = XP2;
% % % XP(3,:) = XP3;

%% Calculate displacement of particles
%Convert XP from cm to mm
XP = XP*10;

[m,N] = size(XP);       %XP is a 3xN matrix where each column contains the (X,Y,Z) components of a point
Xo    = XP(1,:);
Yo    = XP(2,:);
Zo    = XP(3,:);

%Establish location of particles in local R-alpha coordinate system

theta = atan2((Zo-ZR),Yo);               %Defines the "slice" of the vortex ring which affects each particle

R_vec      = zeros(size(XP));       %Defines the position vector with tail at the center of the slice of the vortex ring and head at the particle
R_vec(1,:) = Xo;
R_vec(2,:) = Yo - Ro*cos(theta);
R_vec(3,:) = Zo - ZR - Ro*sin(theta);
R_mag      = sqrt(R_vec(1,:).^2 + R_vec(2,:).^2 + R_vec(3,:).^2);

%Find angle between the R-vector and the Z-plane of the vortex ring
alpha      = asin(Xo./R_mag); 
alpha_temp = alpha;

XP_YZ_mag = sqrt(Yo.^2 + (Zo - ZR).^2);
XG_YZ_mag = Ro;

log     = XP_YZ_mag >= XG_YZ_mag;
log_vec = alpha_temp.*log;
ind     = find(log_vec == 0);

alpha(ind) = pi - alpha_temp(ind);
      

%Move the particles by the amount prescribed by the velocity profile
d      = (8*R_mag*C/L).*exp(-R_mag/L);
alpha2 = alpha + d./R_mag;              %New angular position based on velocity profile

%Convert new positions from R-alpha coordinates to X-Y-Z coordinates
XP2      = zeros(size(XP)); 
XP2(1,:) = R_mag.*sin(alpha2);
XP2(2,:) = Ro.*cos(theta)      + R_mag.*cos(theta).*cos(alpha2);
XP2(3,:) = ZR + Ro.*sin(theta) + R_mag.*sin(theta).*cos(alpha2);

%Convert XP2 from mm to cm
XP2 = XP2/10;













