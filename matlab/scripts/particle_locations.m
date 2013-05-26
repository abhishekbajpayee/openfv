clear all;

[x, y, z] = textread('particle_sim/particles_500.txt');
[xb, yb, zb] = textread('particle_sim/piv_sim_500.txt')

x = x/max(x);
y = y/max(y);
z = z/max(z);
xb = xb/max(xb);
yb = yb/max(yb);
zb = zb/max(zb);

figure;
scatter3(-xb, yb, zb, 'b+');
hold on;
scatter3(x, y, z, 'r+');

%axis([-1,1,-1,1,0,1]);