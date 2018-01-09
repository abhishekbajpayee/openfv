function [result] = export_to_tecplot(field, frames, filename)

result = 0;

step = 3;

% [cx, cy, cz, cav] = curl(field(step).winCtrsX, field(step).winCtrsY, ...
%                          field(step).winCtrsZ, field(step).ux, ...
%                          field(step).uy, field(step).uz);

u = field(step).ux;
v = field(step).uy;
w = field(step).uz;

u = smooth3(u, 'gaussian');
v = smooth3(v, 'gaussian');
w = smooth3(w, 'gaussian');

[cx, cy, cz, cav] = curl(u, v, w);

vorticity = smooth3(real(sqrt(cx.^2 + cy.^2 + cz.^2)));

I = size(field(step).ux, 1);
J = size(field(step).uy, 2);
K = size(field(step).uz, 3);

total_points = I*J*K;

xx = reshape(field(step).winCtrsX, [total_points, 1]);
yy = reshape(field(step).winCtrsY, [total_points, 1]);
zz = reshape(field(step).winCtrsZ, [total_points, 1]);
ux = reshape(field(step).ux, [total_points, 1]);
uy = reshape(field(step).uy, [total_points, 1]);
uz = reshape(field(step).uz, [total_points, 1]);
cx = reshape(cx, [total_points, 1]);
cy = reshape(cy, [total_points, 1]);
cz = reshape(cz, [total_points, 1]);
vor = reshape(vorticity, [total_points, 1]);

header1 = ['TITLE = "' frames(1) '_' frames(2) '"'];
header2 = ['VARIABLES = "X", "Y", "Z", "U", "V", "W", "CX", "CY", ' ...
           '"CZ", "VOR"'];
header3 = ['ZONE T="PIV_thrfixed", SOLUTIONTIME=0, I = ' num2str(I) ', J = ' ...
                    num2str(J) ', K = ' num2str(K) ', DATAPACKING = POINT'];

% Writing Data
f = fopen(filename, 'w');

fprintf(f, '%s\n', header1);
fprintf(f, '%s\n', header2);
fprintf(f, '%s\n', header3);

for i = 1:total_points
    fprintf(f, '   %f', [xx(i) yy(i) zz(i) ux(i) uy(i) uz(i) cx(i) ...
                        cy(i) cz(i) vor(i)]);
    fprintf(f, '\n');
end

result = 1;

end