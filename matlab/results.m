clear all;

[x,y,z, dump] = textread('world_points.txt');
[x y z]
[cx,cy,cz,dump] = textread('camera_points.txt');
%[pa,pb,pc,pd,dump] = textread('plane_params.txt');
%[rt1, rt2, rt3, rt4] = textread('Rt.txt');

ppi = 30;
planes = length(x)/30;

%[bx,by] = meshgrid(-10:10);
%[bx,by] = meshgrid([ccx-100:10:ccx+100],[ccy-100:10:ccy+100]);

figure;
hold on;
for i=1:2%planes
    scatter3(x((i-1)*ppi+1:i*ppi),y((i-1)*ppi+1:i*ppi),z((i-1)*ppi+1:i*ppi),'b+');
    %bz = (-pd(i)-pa(i)*bx-pb(i)*by)/pc(i);
    %mesh(bx,by,bz,'EdgeColor','black');
    %hold off;
end

%plot3(cx,cy,cz);

plot3([0 0], [0 0], [0 100], 'r');
plot3([0 0], [0 100], [0 0]);
plot3([0 100], [0 0], [0 0], 'b');

axis equal;
