x =    [10 50 100 250 500  750  1000];
t50 =  [19 97 202 515 1248 2117 4036];
t90 =  [9  49 99  237 494  619  1005];
t100 = [9  47 90  224 449  460  917];

plot(x,x,'-s',x,t50,'-s',x,t90,'-s',x,t100,'-s');

legend('Actual Number',...
       'Threshold = 50',...
       'Threshold = 90',...
       'Threshold = 100');

xlabel('Seeded Particles [ ]');
ylabel('Detected Particles [ ]');
title('Detected Particles vs. Seeded Particles for different threshold levels');
axis([0,1000,0,1000]);