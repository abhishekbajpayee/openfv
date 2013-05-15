

%[z, cx_100] = textread('zres.txt');

plot(z,cx,z,cx_20,z,cx_40,z,cx_60,z,cx_80,z,cx_100);
legend('Threshold = 0','Threshold = 20','Threshold = 40','Threshold = 60','Threshold = 80','Threshold = 100');
xlabel('Refocusing depth z [mm]');
ylabel('Cross Correlation [ ]');
title('Cross Correlation of image refocusing at z with image refocused at z = 5.0 mm'); 