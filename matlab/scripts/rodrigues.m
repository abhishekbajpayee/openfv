function[R] = rodrigues(r)

theta = norm(r);
r = r/theta;

I = eye(3);

R = cos(theta)*I + ((1-cos(theta))*(r'*r)) + sin(theta)*[0 -r(3) r(2);r(3) 0 -r(1);-r(2) r(1) 0];

end