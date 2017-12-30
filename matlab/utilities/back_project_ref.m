function [pA, pB] = back_project_ref(z, cxy, XC, P, ref)

dist = @(x) ref_project_dist([x(1), x(2)], z, cxy, XC, P, ref);

pA = fminsearch(dist, [0,0]);
pA(3) = z;

[tempA, pB, temp] = img_refrac(XC, pA', ref.geom(1), ref.geom(2), ...
                               ref.geom(3:end));
