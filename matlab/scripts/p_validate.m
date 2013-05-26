clear all;

[col1, col2, col3, col4, dump] = textread('P_mats.txt');
[rt1, rt2, rt3, rt4] = textread('Rt.txt');
<<<<<<< HEAD
[art1, art2, art3, art4] = textread('aligned_rt.txt');
[x,y,z, dump] = textread('world_points.txt');
[f] = textread('f.txt');
[align_data] = textread('../ba_files/scene_align.txt');

% cams = length(col1)/3;
% 
% for i=1:cams
%     P(:,1,i) = col1((3*(i-1))+1:(3*(i-1))+3);
%     P(:,2,i) = col2((3*(i-1))+1:(3*(i-1))+3);
%     P(:,3,i) = col3((3*(i-1))+1:(3*(i-1))+3);
%     P(:,4,i) = col4((3*(i-1))+1:(3*(i-1))+3);
%     Rt(:,1,i) = rt1((3*(i-1))+1:(3*(i-1))+3);
%     Rt(:,2,i) = rt2((3*(i-1))+1:(3*(i-1))+3);
%     Rt(:,3,i) = rt3((3*(i-1))+1:(3*(i-1))+3);
%     Rt(:,4,i) = rt4((3*(i-1))+1:(3*(i-1))+3);
% end

K = zeros(3);
=======
[x,y,z, dump] = textread('world_points.txt');
[f] = textread('f.txt');

cams = length(col1)/3;

for i=1:cams
    P(:,1,i) = col1((3*(i-1))+1:(3*(i-1))+3);
    P(:,2,i) = col2((3*(i-1))+1:(3*(i-1))+3);
    P(:,3,i) = col3((3*(i-1))+1:(3*(i-1))+3);
    P(:,4,i) = col4((3*(i-1))+1:(3*(i-1))+3);
    Rt(:,1,i) = rt1((3*(i-1))+1:(3*(i-1))+3);
    Rt(:,2,i) = rt2((3*(i-1))+1:(3*(i-1))+3);
    Rt(:,3,i) = rt3((3*(i-1))+1:(3*(i-1))+3);
    Rt(:,4,i) = rt4((3*(i-1))+1:(3*(i-1))+3);
end

figure;
hold on;

K = zeros(3);
K(1,1) = f(1);
K(2,2) = f(1);
>>>>>>> origin/master
K(1,3) = 646;
K(2,3) = 482;
K(3,3) = 1;

<<<<<<< HEAD
%%

ix = [0 646*2 0 646*2 646
      0 0 482*2 482*2 482
      1 1 1 1 1];

figure; hold on;
  
for k=1:9

    ind = (k-1)*3;
    R = [rt1(ind+1:ind+3) rt2(ind+1:ind+3) rt3(ind+1:ind+3)];
    t = rt4(ind+1:ind+3);
    C_ua(k,:) = -R'*t;
    
    K(1,1) = f(k);
    K(2,2) = f(k);
    M = K*R;
    Minv = inv(M);
    p4 = K*t;
    C = -Minv*p4;
    %scatter3(C(1), C(2), C(3), 'b+');

    for j=1:5

        line = [];
        for i=0:100:1000
            X = Minv*(i*ix(:,j) - p4);
            line = [line X];
        end

        %plot3(line(1,:),line(2,:),line(3,:),'y');

    end

end

for k=1:9

    ind = (k-1)*3;
    R = [art1(ind+1:ind+3) art2(ind+1:ind+3) art3(ind+1:ind+3)];
    t = art4(ind+1:ind+3);
    C_a(k,:) = -R'*t;
    
    K(1,1) = f(k);
    K(2,2) = f(k);
    M = K*R;
    Minv = inv(M);
    p4 = K*t;
    P = [M p4];
    C = -Minv*p4;
    %scatter3(C(1), C(2), C(3), 'b+');

    for j=1:5

        line = [];3
        for i=0:100:1000
            X = Minv*(i*ix(:,j) - p4);
            line = [line X];
        end

        %plot3(line(1,:),line(2,:),line(3,:),'g');

    end

end

[px,py] = meshgrid(-200:10:200);
pz = px*0;
%mesh(px,py,pz);

axis equal;

plot3(C_ua(:,1),C_ua(:,2),C_ua(:,3),'b');
plot3(C_a(:,1),C_a(:,2),C_a(:,3),'r');

rm = [rt1(1:3),rt2(1:3),rt3(1:3)];
for i=2:9
    ind = (i-1)*3;
    rtmp = [rt1(ind+1:ind+3),rt2(ind+1:ind+3),rt3(ind+1:ind+3)];
    rm = rm + rtmp;
end
rm = rm/9;

C_r = rm*C_ua';
plot3(C_r(1,:),C_r(2,:),C_r(3,:),'y');
=======
for i=1:30
    pred = P(:,:,1)*[x(i);y(i);z(i);1];
    u = pred(1)/pred(3);
    v = pred(2)/pred(3);
    plot(u,v,'b+');
    
    pred = Rt(:,:,1)*[x(i);y(i);z(i);1];
    u = pred(1)/pred(3);
    v = pred(2)/pred(3);
    u = f(1)*u + 646;
    v = f(1)*v + 482;
    plot(u,v,'r+');
    
    P_tmp = K*Rt(:,:,1);
    pred = P_tmp*[x(i);y(i);z(i);1];
    u = pred(1)/pred(3);
    v = pred(2)/pred(3);
    plot(u,v,'g+');
end
>>>>>>> origin/master
