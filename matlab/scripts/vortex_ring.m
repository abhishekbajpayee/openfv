%% vortex ring

clear all;

write = 0;
visualize = 1;

XP = [];

size = 150;
num = 250;

for i=1:num
    
    x = (rand()*size)-size*0.5;
    y = (rand()*size)-size*0.5;
    z = (rand()*size)-size*0.5;
    
    point = [x;y;z];
    XP = [XP point];
    
end

%figure;
%scatter3(XP(1,:),XP(2,:),XP(3,:),'.');

R = (size/4); % radius of ring
zplane = -size*0.4; % z plane of ring (can move if ring moves)
r = (R*2/5); % radius of circulation ring
vz = size*0.8;
time = 1; % in seconds
fps = 30;
frames = fps*time;
dt = 1/fps;
T = 100;

if (write)
    file = fopen('vortex_points_rare.txt','w');
end

for i=1:frames
    
    zplane = zplane + vz*dt;
    XP2 = synthetic_3d_vort_ring_generator(XP,R,zplane,T,r,dt);

    distance = sqrt( (XP2(1,:)-XP(1,:))*(XP2(1,:)-XP(1,:))' + ...
               (XP2(2,:)-XP(2,:))*(XP2(2,:)-XP(2,:))' + ...
               (XP2(3,:)-XP(3,:))*(XP2(3,:)-XP(3,:))' );
           
    max(distance)
    
    XP = XP2;
    
    if (write)
        for j=1:num
            fprintf(file,'%f\t%f\t%f\n',XP2(1,j),XP2(2,j),XP2(3,j));
        end
    end
    
    if (visualize)
        scatter3(XP2(1,:),XP2(2,:),XP2(3,:),'.');
        axis([-size*0.5,size*0.5,-size*0.5,size*0.5,-size*0.5,size*0.5]);
        M(i) = getframe;
    end
    
    %T = T*0.98;
    %vz = vz*0.5
    
end

if (write)
    fclose(file);
end

if (visualize)
    movie(M,10,fps);
end