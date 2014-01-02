function Ivol = project_intensity(locations,h,method)

% This function takes a kernel representing the intensity profile of a
% particle and inserts it at locations in a synthetic volume. Inputs are
% point cloud of particle locations and 3D kernel(x,y,z) to use.

method = 'gaussian';

pixperphys = 9.533433;
imsz_u = 1280;
imsz_v = 800;
deltaz = 1.0;
zoffset = 0;
zmin = -55.0; % minimum and maximum in the refocused coordinate frame
zmax = 55.0;
z_range = zmax - zmin;
z_locations = [zmin:deltaz:zmax];

% hsize = size(h);
% hcenter = round(hsize/2);
% hrange = hsize-hcenter;

% convert world coordinates to pixel
x = locations(:,1);
y = locations(:,2);
z = locations(:,3);

xpix = x*pixperphys + imsz_u/2;
ypix = y*pixperphys + imsz_v/2;
zpix = (z-zmin)/deltaz+1;

X_im = [ypix xpix zpix];

Ivol = zeros(imsz_v,imsz_u,length(z_locations));

% This method inserts the kernel center at the nearest voxel location and
% can be used for arbitrary kernel profiles
if strcmp(method,'insert')

    X_im(:,1)=round(X_im(:,1));
    X_im(:,2)=round(X_im(:,2));
    X_im(:,3)=round(X_im(:,3));

    % remove out of range values
    X_im = X_im(X_im(:,1)>0,:);
    X_im = X_im(X_im(:,2)>0,:);
    X_im = X_im(X_im(:,3)>0,:);

    for i = 1:length(X_im);    
        Ybegin = X_im(i,1)-hrange(1);
        Yend = X_im(i,1)+hrange(1);
        Xbegin = X_im(i,2)-hrange(2);
        Xend = X_im(i,2)+hrange(2);
        Zbegin = X_im(i,3)-hrange(3);
        Zend = X_im(i,3)+hrange(3);
                     
        Ivol(Ybegin:Yend,Xbegin:Xend,Zbegin:Zend)=h;
    end
    
end

if strcmp(method,'gaussian');

% This method includes subvoxel accuracy for the kernel center using a gaussian model for the particle
    hsize = 5;
    hrange = (hsize-1)/2;
    hcenter = round(hsize/2);
    sig = hsize/(4*sqrt(2*log(2)));
    

    for i = 1:length(z)
        Ybegin = max(floor(X_im(i,1)-hrange),1);
        Yend = min(ceil(X_im(i,1)+hrange),imsz_v);
        Xbegin = max(floor(X_im(i,2)-hrange),1);
        Xend = min(ceil(X_im(i,2)+hrange),imsz_u);
        Zbegin = min(floor(X_im(i,3)-hrange),1);
        Zend = max(ceil(X_im(i,3)+hrange),length(z_locations));

%         Ybegin = Ybegin(Ybegin>0);
%         Xbegin = Xbegin(Xbegin>0);
%         Zbegin = Zbegin(Zbegin>0);
% 
%         Yend = Yend(Yend<=imsz_v);
%         Xend = Xend(Xend<=imsz_u);
%         Zend = Zend(Zend<=length);

        dy = (Ybegin:Yend)-X_im(i,1);
        dx = (Xbegin:Xend)-X_im(i,2);
        dz = (Zbegin:Zend)-X_im(i,3);

        [DY,DX,DZ]=ndgrid(dy,dx,dz);

        h = exp(-(DY.*DY/2/sig^2 + DX.*DX/2/sig^2 + DZ.*DZ/2/sig^2));

        Ivol(Ybegin:Yend,Xbegin:Xend,Zbegin:Zend)=h;
    end
end



