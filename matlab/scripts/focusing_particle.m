path = '/home/ab9/projects/cropped/mult/';
%path = '/home/ab9/projects/cropped/add/thresh_50/';
%path = '/home/ab9/projects/cropped/add/thresh_100/';
%path = '/home/ab9/projects/cropped/add/thresh_150/';

%figure;
count=1;
for i=-1.5:0.5:1.5
    filename = sprintf('z_%.2f.jpg',i);
    I = imread([path filename]);
    subplot(4,7,count+0);
    image(I);
    %title(filename);
    axis([1,10,1,10]);
    %axis equal;
    colormap gray;
    count = count+1;
end
