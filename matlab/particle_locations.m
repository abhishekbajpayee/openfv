[x, y, z] = textread('particles.txt');
[xb, yb, zb] = textread('piv_sim_500.txt');

xo = mean(x);
yo = mean(y);
zo = mean(z);
x = x-xo;
y = y-yo;
z = z-29.5;

xo = mean(xb);
yo = mean(yb);
zo = mean(zb);
xb = xb-xo;
yb = yb-yo;
zb = zb+1000;

figure;
scatter3(xb, yb, zb, 'b+');
hold on;
scatter3(x, -y, -z, 'r+');

axis equal;

view(0,90);

%%

%%