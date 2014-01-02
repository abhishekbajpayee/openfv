% This function compares reconstructed intensity volumes to synthetic
% intensity volumes to determine reconstruction quality Q
% Written by Leah Mendelson 4/3/13
% 
path = '../../../experiment/thesis/test_field_1000/';
locations = textread([path 'config.txt']);
Ivol = project_intensity(locations,0,'gaussian');

%direc = uigetdir;
direc = [path 'vol_4p'];
imnames = dir([direc '/*.jpg']);
imnames = {imnames.name};

imnames = sort_nat(imnames);

im1 = zeros(800,1280,length(imnames));

for i = 1:length(imnames);
    itemp = imread([direc '/' imnames{i}]);
    itemp = im2double(itemp);
    for j=2:800
        for k=2:1280
            im1(j,k,i)=itemp(j-1,k-1);
        end
    end
end

im2 = Ivol;

%enable this section to threshold before computing Q

Qn_tot = 0;
Qd1_tot = 0;
Qd2_tot = 0;

%for i = 1:5;
for i = 1:111;
    Qnum = im1(:,:,i).*im2(:,:,i);
    Qnum = sum(sum(Qnum));
    Qn_tot = Qn_tot+Qnum;
    
    Qd1 = im1(:,:,i).^2;
    Qd1 = sum(sum(Qd1));
    Qd1_tot = Qd1_tot+Qd1;
    
    Qd2 = im2(:,:,i).^2;
    Qd2 = sum(sum(Qd2));
    Qd2_tot = Qd2_tot+Qd2;
      
end

Qd = sqrt(Qd1_tot*Qd2_tot);
Q = Qn_tot/Qd