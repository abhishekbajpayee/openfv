function [result] = export_to_tecplot(field, frames, filename)

result = 0;

step = 2;
I = size(field(step).ux, 1);
J = size(field(step).ux, 2);
K = size(field(step).ux, 3);

total_points = I*J*K;

header1 = ['TITLE = "' frames(1) '_' frames(2) '"'];
header2 = ['VARIABLES = "X", "Y", "Z", "U", "V", "W"'];
header3 = ['ZONE T="PIV_thrfixed", SOLUTIONTIME=0, I = ' num2str(I) ', J = ' ...
                    num2str(J) ', K = ' num2str(K) ', DATAPACKING = POINT'];

xx = reshape(field(step).winCtrsX, [total_points, 1]);
yy = reshape(field(step).winCtrsY, [total_points, 1]);
zz = reshape(field(step).winCtrsZ, [total_points, 1]);
ux = reshape(field(step).ux, [total_points, 1]);
uy = reshape(field(step).uy, [total_points, 1]);
uz = reshape(field(step).uz, [total_points, 1]);

f = fopen(filename, 'w');

fprintf(f, '%s\n', header1);
fprintf(f, '%s\n', header2);
fprintf(f, '%s\n', header3);

for i = 1:total_points
    fprintf(f, '   %f', [xx(i) yy(i) zz(i) ux(i) uy(i) uz(i)]);
    fprintf(f, '\n');
end

end