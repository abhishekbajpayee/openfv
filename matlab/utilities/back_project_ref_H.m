% Back projects points to a given z based on the supplied P matrix
% and taking refractive geometry defined in ref into account

function [proj1, proj2] = back_project_ref_H(pts, z, XC, P, ref)

[H1, H2] = find_back_project_H(z, XC, P, ref);

matrix = 0;

s = size(pts);
ss = size(s);
if ss(2) == 3
    matrix = 1;
end
if s(1) ~= 3
    display('Dat first dimension must be 3 yo!');
end

if matrix
    pts = reshape(pts, [3, s(2)*s(3)]);
else
    error('unsupported if incoming array not matrix');
end

% calculating back projected points lying at z

proj1 = H1*pts;

q = zeros(size(proj1));
q(1,:) = proj1(3,:);
q(2,:) = proj1(3,:);
q(3,:) = proj1(3,:);

proj1 = proj1./q;
proj1(3,:) = z;

% calculating back projected points just inside
% glass wall

proj2 = H2*pts;

q = zeros(size(proj2));
q(1,:) = proj2(3,:);
q(2,:) = proj2(3,:);
q(3,:) = proj2(3,:);

proj2 = proj2./q;
proj2(3,:) = ref.geom(1)+ref.geom(2);

end