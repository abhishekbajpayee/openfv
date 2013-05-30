clear all;

[x, y, z] = textread('../data_files/particle_sim/particles_grid.txt');
[xm, ym, zm] = textread('../data_files/particle_sim/particles_grid_mult.txt');
[xb, yb, zb] = textread('../data_files/particle_sim/piv_sim_grid.txt');

x = x/1.2202;
y = y/1.2202;

xm = xm/1.2202;
ym = ym/1.2202;

ind1 = 71;
ind2 = 69;
x = x-x(ind1);
y = y-y(ind1);
z = z-z(ind1);
xm = xm-xm(ind2);
ym = ym-ym(ind2);
zm = zm-zm(ind2);

X = [x, y, z];
Xm = [xm, ym, zm];
XB = [xb, yb, zb];

%x = x/max(x);
%y = y/max(y);
%z = z/max(z);
%xb = xb/max(xb);
%yb = yb/max(yb);
%zb = zb/max(zb);

figure;
scatter3(xb, yb, zb, 'b+');
hold on;
scatter3(xm, ym, zm, 'r+');

%axis([-1,1,-1,1,0,1]);

xpa = [];
ypa = [];
zpa = [];
xpm = [];
ypm = [];
zpm = [];
adt = [];
adx = [];
ady = [];
adz = [];
mdt = [];
mdx = [];
mdy = [];
mdz = [];
ad = [];
md = [];
tdista=0;
tdistm=0;

for i=1:length(x)
    for j=1:length(xb)
        
        dista = sqrt( (x(i)-xb(j))^2 + (y(i)-yb(j))^2 + (z(i)-zb(j))^2 );
        distm = sqrt( (xm(i)-xb(j))^2 + (ym(i)-yb(j))^2 + (zm(i)-zb(j))^2 );
        
        if (dista<1)
            tdista=tdista+dista;
            ad = [ad dista];
            xpa = [xpa x(i)];
            ypa = [ypa y(i)];
            zpa = [zpa z(i)];
            adt = [adt dista];
            adx = [adx x(i)-xb(j)];
            ady = [ady y(i)-yb(j)];
            adz = [adz z(i)-zb(j)];
        end
        
        if (distm<1)
            tdistm=tdistm+distm;
            md = [md distm];
            xpm = [xpm x(i)];
            ypm = [ypm y(i)];
            zpm = [zpm z(i)];
            mdt = [mdt distm];
            mdx = [mdx xm(i)-xb(j)];
            mdy = [mdy ym(i)-yb(j)];
            mdz = [mdz zm(i)-zb(j)];
        end
        
    end
end

tdista
tdistm
tdista/length(x)
tdistm/length(xm)

figure;
scatter(xpa, adx, 'b.');
hold on;
scatter(xpm, mdx, 'r.');
xlabel('x');

figure;
scatter(ypa, ady, 'b.');
hold on;
scatter(ypm, mdy, 'r.');
xlabel('y');

figure;
scatter(zpa, adz, 'b.');
hold on;
scatter(zpm, mdz, 'r.');
xlabel('z');

figure;
xax = [1:1:125];
plot(xax,ad,xax,md);