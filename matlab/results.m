clear all;

[x,y,z, dump] = textread('world_points.txt');
[cx,cy,cz,dump] = textread('camera_points.txt');
[pa,pb,pc,pd,dump] = textread('plane_params.txt');
<<<<<<< HEAD
[rt1, rt2, rt3, rt4] = textread('Rt.txt');

r = [rt1(1:3),rt2(1:3),rt3(1:3)];
for i=2:9
    ind = (i-1)*3;
    rtmp = [rt1(ind+1:ind+3),rt2(ind+1:ind+3),rt3(ind+1:ind+3)];
    r = r + rtmp;
end
r = r/9
=======
>>>>>>> origin/master

ppi = 30;
planes = length(x)/30;

<<<<<<< HEAD
[bx,by] = 3meshgrid(-10:10);

% point cloud centroid
%cx = mean(x);
%cy = mean(y);
%cz = mean(z);
=======
[bx,by] = meshgrid(-10:10);

% point cloud centroid
cx = mean(x);
cy = mean(y);
cz = mean(z);
>>>>>>> origin/master

% camera center centroid
ccx = mean(cx);
ccy = mean(cy);
ccz = mean(cz);

<<<<<<< HEAD
[bx,by] = meshgrid([ccx-100:10:ccx+100],[ccy-100:10:ccy+100]);
=======



>>>>>>> origin/master

figure;
hold on;
for i=1:planes
<<<<<<< HEAD
    %scatter3(x((i-1)*ppi+1:i*ppi),y((i-1)*ppi+1:i*ppi),z((i-1)*ppi+1:i*ppi),'b+');
=======
    scatter3(x((i-1)*ppi+1:i*ppi),y((i-1)*ppi+1:i*ppi),z((i-1)*ppi+1:i*ppi),'b+');
>>>>>>> origin/master
    %bz = (-pd(i)-pa(i)*bx-pb(i)*by)/pc(i);
    %mesh(bx,by,bz,'EdgeColor','black');
    %hold off;
end

<<<<<<< HEAD
for i=1:60%length(x)
    scatter3(x(i),y(i),z(i),'b+');
    rotated = r'*[x(i);y(i);z(i)];
    scatter3(rotated(1),rotated(2),rotated(3),'r+');
end

%bz = (5486.87-2.40807*bx-2.7708*by)/-5.1163;
%mesh(bx,by,bz,'EdgeColor','black');

%scatter3(cx,cy,cz,'+');
=======
scatter3(cx,cy,cz,'+');
>>>>>>> origin/master

plot3([0 0], [0 0], [0 100], 'r');
plot3([0 0], [0 100], [0 0]);
plot3([0 100], [0 0], [0 0], 'b');

<<<<<<< HEAD
axis equal;
=======
axis equal;
>>>>>>> origin/master
