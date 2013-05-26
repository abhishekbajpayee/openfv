clear all;

params = textread('../ba_files/scene_align.txt');

for i=1:9
    for j=1:3
        r(i,j) = params(1+((i-1)*6)+j);
        t(i,j) = params(1+((i-1)*6)+j+3);
    end
end

for i=1:9
    R(:,:,i) = rodrigues(r(i,:));
end

for i=1:9
    X(i,:) = -R(:,:,i)'*t(i,:)';
end

% figure;
% scatter3(X(:,1),X(:,2),X(:,3));
% hold on;
% scatter3(0,0,0,'r+');
% plot3([0 0], [0 0], [0 100], 'r');
% plot3([0 0], [0 100], [0 0]);
% plot3([0 100], [0 0], [0 0]);
% axis equal;

r_c = mean(r)
t_c = mean(t)
r
t