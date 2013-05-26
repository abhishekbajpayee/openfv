clear all;

path = '../cropped/thresh_150/';

fcount = 1;

figure;
hold on;

for i=-1.5:0.1:1.5

    str = num2str(i,'z_%.2f.jpg');
    depth = num2str(i,'z = %.2f');
    fullpath = [path str];
    I = imread(fullpath);
    imshow(I, 'InitialMagnification', 5000);
    text(1, 1, i, ['z = ',num2str(i)],...
         'HorizontalAlignment','left',...
         'Color', [1,1,1], 'FontSize', 15);
    
    %colorbar;
    caxis([0,255]);
    axis image;
    %title(depth);
    
    M(fcount) = getframe;
    fcount = fcount+1;

end

movie(M,10,10);
%writer = VideoWriter('movie.avi','Motion JPEG AVI');
%open(writer);
%writeVideo(writer,M);
%close(writer);
