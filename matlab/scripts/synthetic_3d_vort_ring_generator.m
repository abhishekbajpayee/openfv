%   Written by Jesse Belden 
%   jbelden@mit.edu
%   March 31, 2009
%
% This program simulates the motion induced by a theoretical vortex ring as
% presented in Elsinga et al. (2005).  XP is a (3,N) matrix of point
% locations (in cm!).  Ro is the radius of the vortex ring (in mm!). L is a
% lengthscale defining the size of the core of the ring (in mm!), ZR is the
% z-offset of the ring (in mm!).  C is the conversion factor of the depth
% dimension of a voxel from physical space to voxel units (vox/mm).

function [XP2] = synthetic_3d_vort_ring_generator(XP,Ro,ZR,T,L,dt)


% R  = 0:0.01:10;  %Radial distance of point from center of vortex ring
% d  = (8*R*C/L).*exp(-R/L);

% % % %TEST PARTICLES TO MAKE SURE CODE IS WORKING
% % % ang = 0;
% % % XP1 = zeros(1,11) + Ro*sin(ang);
% % % XP2 = zeros(1,11) + Ro*cos(ang);
% % % XP3 = 0.1:1:10.1;disp('playing');
% % % 
% % % % %Change angle
% % % % for j = 1:7XP(1,:)
% % % %     ang = ang + pi/4;5
% % % %     XP1 = [XP1 zeros(1,11) + Ro*sin(ang)];
% % % %     XP2 = [XP2 zeros(1,11) + Ro*cos(ang)];
% % % %     XP3 = [XP3 0.1:1:10.1];
% % % % end
% % % 
% % % %Change distance
% % % A = Ro;
% % % for j = 1:11
% % %     A = A-2;25
% % %     XP1 = [XP1 zeros(1,11) + A*sin(ang)];
% % %     XP2 = [XP2 zeros(1,11) + A*cos(ang)];
% % %     XP3 = [XP3 0.1:1:10.1];
% % % end
% % % XP(1,:) = XP1; 
% % % XP(2,:) = XP2;
% % % XP(3,:) = XP3;

%% Calculate displacement of particles

[m,N] = size(XP);       %XP is a 3xN matrix where each column contains the (X,Y,Z) components of a point
Xo    = XP(1,:);
Yo    = XP(2,:);
Zo    = XP(3,:);

%Establish location of particles in local R-alpha coordinate system

theta = atan2(Yo,Xo);               %Defines the "slice" of the vortex ring which affects each particle

R_vec      = zeros(size(XP));       %Defines the position 10vector with tail at the center of the slice of the vortex ring and head at the particle
R_vec(1,:) = Xo - Ro*cos(theta);
R_vec(2,:) = Yo - Ro*sin(theta);
R_vec(3,:) = Zo - ZR;
R_mag      = sqrt(R_vec(1,:).^2 + R_vec(2,:).^2 + R_vec(3,:).^2);


%Find angle between the R-vector and the Z-plane of the vortex ring
alpha      = asin((Zo-ZR)./R_mag); 
alpha_temp = alpha;

XP_XY_mag = sqrt(Xo.^2 + Yo.^2);
XG_XY_mag = sqrt((Ro*cos(theta)).^2 + (Ro*sin(theta)).^2);

log     = XP_XY_mag >= XG_XY_mag;
log_vec = alpha_temp.*log;
ind     = find(log_vec == 0);

alpha(ind) = pi - alpha_temp(ind);    

%Move the particles by the amount prescribed by the velocity profile

d      = dt*(R_mag*(T/L)).*exp(-R_mag/L);
alpha2 = alpha + d./R_mag;              %New angular position based on velocity profile

%Convert new positions from R-alpha coordinates to X-Y-Z coordinates
XP2      = zeros(size(XP)); 
XP2(1,:) = Ro.*cos(theta) + R_mag.*cos(theta).*cos(alpha2);
XP2(2,:) = Ro.*sin(theta) + R_mag.*sin(theta).*cos(alpha2);
XP2(3,:) = ZR             + R_mag.*sin(alpha2);











