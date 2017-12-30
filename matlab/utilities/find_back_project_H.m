function [H1, H2] = find_back_project_H(z, XC, P, ref)

src_pts = [-10 -10;10 -10;-10 10;10 10];
dst_pts = [];
inside_pts = [];

for i = 1:size(src_pts, 1)
    [dst_p, inside_p] = back_project_ref(z, src_pts(i,:), XC, P, ...
                                         ref);
    dst_pts = [dst_pts; dst_p];
    inside_pts = [inside_pts; inside_p(1:3)'];
end

H1 = findHomography(src_pts', dst_pts(:,1:2)');
H2 = findHomography(src_pts', inside_pts(:,1:2)');

end