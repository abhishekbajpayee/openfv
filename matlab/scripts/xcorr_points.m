clear all;

path = '/home/ab9/projects/experiment/thesis/test_field_1000/';
X = textread([path 'config.txt']);

Xf = textread([path 'particles/4p/f_all_w2_t40_c15.txt'], '', 'headerlines', 2);

Xf(:,3) = Xf(:,3)-50;

ref = Xf;
cdist=0; tol = 0.5;
n=0;
xdist = [];
ydist = [];
zdist = [];

for i=1:length(X(:,1))
    for j=1:length(ref(:,1))
        p1 = X(i,:);
        p2 = ref(j,:);
        d = p1-p2;
        if (norm(d*d')<tol)
            cdist=cdist+norm(d*d');
            xdist = [xdist; p2(1) d(1)];
            ydist = [ydist; p2(2) d(2)];
            zdist = [zdist; p2(3) d(3)];
            n=n+1;
        end
    end
end

plots = 0;
if (plots)
    
    figure;
    hold on;
    scatter3(X(:,1),X(:,2),X(:,3),'+');
    scatter3(ref(:,1),ref(:,2),ref(:,3));

    figure;
    scatter(xdist(:,1), xdist(:,2));
    figure;
    scatter(ydist(:,1), ydist(:,2));
    figure;
    scatter(zdist(:,1), zdist(:,2));

end

fprintf('Summary\n');
fprintf('mError: %f\n', cdist/n);
fprintf('xError: %f\n', mean(xdist(:,2)));
fprintf('yError: %f\n', mean(ydist(:,2)));
fprintf('zError: %f\n\n', mean(zdist(:,2)));