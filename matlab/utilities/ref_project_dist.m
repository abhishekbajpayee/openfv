function [dist] = ref_project_dist(pxy, z, cxy, XC, P, ref)

pt = img_refrac(XC, [pxy(1); pxy(2); z], ref.geom(1), ref.geom(2), ...
                ref.geom(3:end));

proj = P*pt;
proj(1) = proj(1)/proj(3);
proj(2) = proj(2)/proj(3);

dist = (cxy(1) - proj(1))^2 + (cxy(2) - proj(2))^2;

end