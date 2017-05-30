function hammingFigure
% Produces the picture demonstrating the effect of hamming and blackman
% windows have on the problem of 3D cardinal interpolation

%   Copyright (c) 2007-2015  Thomas H. Clark
% Coarse grid of points containing a gaussian pulse, sd 1 voxel, as
% typified by a particle or narrow correlation peak in Tomographic PIV.
sincTestArray = positionGaussian(zeros(48,48,48), [9 9 9], 1);
sincTestArray = sincTestArray/(max(sincTestArray(:)));
% x = linspace(1,48,size(sincTestArray,1));
% [X Y Z] = ndgrid(x,x,x);

% We'll also want to plot the sampling points
[Xplot Yplot Zplot] = ndgrid(8:10,8:10,8:10);

% 3D filters are derived (effectively) by taking the outer product of 1D
% filters. Using the alternative rotation method produces similar (although
% not identical) results.

% We interpolate to a window of dimension 40x40x40 in the 2x2x2 voxel
% region surrounding the peak. We do 9 tests of N = 5,7,9, each with no
% filter, a Hamming window and a Blackman window.
% See refs:
%       Scarano 2002, Niblack 1986 (ref in scarano), Stearns and hush. Note
%       that the derivation in stearns and hush is low pass filtered
%       already...??
wSize = [40 40 40];
x2 = linspace(8,10,wSize(1));
[X2 Y2 Z2] = ndgrid(x2,x2,x2);
raiseFigure('3D Whittaker Interpolation Windowing Test');
clf; hold on
ctr = 1;
% Sampling rates
nSamples = [5 7 9];
for iN = 1:3
    
    % Sampling rate
    N = nSamples(iN);
    
    % Get filters for this N
    i = 0:(N-1);
    w_non = ones(size(i));
    w_ham = -0.46*cos(2*pi*i/(N-1))+0.5;
    w_blk = -0.5*cos(2*pi*i/(N-1))+0.08*cos(4*pi*i/(N-1))+0.42;
    
    for filter = 1:3
        
        % Select the window to apply to the truncated sinc expansion
        if filter == 1
            wstr = 'No windowing, ';
            w = w_non;
        elseif filter == 2
            wstr = '3D Hamming window, ';
            w = w_ham;
        elseif filter == 3
            wstr = '3D Blackman window, ';
            w = w_blk;
        end
        
        % Interpolate the field to get the window
        window = whitInterp3_windowed(sincTestArray, wSize, N, w);
        
        % Plot into a subplot
        subplot(numel(nSamples),3,ctr)
        ph3 = plot3(Xplot(:),Yplot(:),Zplot(:),'r.');
        set(ph3,'markerSize',8)
        ph2 = patch(isosurface(X2,Y2,Z2,window,      0.65));
        hold on
        set(ph2,'faceColor','b','faceAlpha',1,'edgeAlpha',0.3)
        camlight right
        lighting phong
        axis equal
        xlim([7.5 10.5])
        ylim([7.5 10.5])
        zlim([7.5 10.5])
        title([wstr 'nSamples = ' num2str(N)])
        
        % Increment the subplot counter
        disp(['Updated subplot ' num2str(ctr) ' of 9'])
        ctr = ctr+1;
        

    end % endfor filter
end % endfor iN



% SUBROUTINE TO ADD A GAUSSIAN PEAK TO THE INTENSITY ARRAY
function intensity = positionGaussian(intensity, position, sigma)

    % Distance from the gaussian position
    [coord1 coord2 coord3] = ndgrid(1:size(intensity,1),1:size(intensity,2), 1:size(intensity,3));
    coord1 = coord1 - position(1);
    coord2 = coord2 - position(2);
    coord3 = coord3 - position(3);
    radius = sqrt(coord1.^2 + coord2.^2 + coord3.^2);
    
    % Gaussian Function:
    intensity = intensity + exp(-0.5 * (radius./sigma).^2) ./ (sqrt(2*pi) .* sigma);

% SUBROUTINE TO DO WINDOW DEFORMATION IN MATLAB
function [window] = whitInterp3_windowed(field, wSize, N, w1)


% Get window element indices in a list form
rVec = linspace(8,10,wSize(1));
cVec = linspace(8,10,wSize(1));
pVec = linspace(8,10,wSize(1));
[RVol CVol PVol] = ndgrid(rVec,cVec,pVec);
pMap = [RVol(:) CVol(:) PVol(:)];




% Get their integer positions
pMapRound = round(pMap);
offset = floor(N/2);

% And their local coord frame equivalents (location in local NxNxN frame)
tMap = pMap-pMapRound;


% Preallocate the window
window = zeros(wSize);


% For each element
for iElem = 1:size(pMap,1);
    
    % Retrieve the 11^3 array surrounding the interpolation point: Amplitude at
    % sampling locations is contained in f_t
    row = pMapRound(iElem,1);
    col = pMapRound(iElem,2);
    dep = pMapRound(iElem,3);
    f_t = field(row-offset:row+offset,col-offset:col+offset,dep-offset:dep+offset);
    amplitude = 0;
    for k = -offset:offset
        for j = -offset:offset
            for i = -offset:offset
                
                inds = [i j k]+offset+1;
                E = f_t(inds(1),inds(2), inds(3));
                
                iMinusx = pi*(i - tMap(iElem,1));
                jMinusy = pi*(j - tMap(iElem,2));
                kMinusz = pi*(k - tMap(iElem,3));
                multx = w1(i+offset+1)*sin(iMinusx)/iMinusx;
                multy = w1(j+offset+1)*sin(jMinusy)/jMinusy;
                multz = w1(k+offset+1)*sin(kMinusz)/kMinusz;
                if isinf(multx) || isnan(multx)
                    multx = 1;
                end
                if isinf(multy) || isnan(multy)
                    multy = 1;
                end
                if isinf(multz) || isnan(multz)
                    multz = 1;
                end
                
                amplitude = amplitude+(multx*multy*multz*E);
            end
        end
    end
    window(iElem) = amplitude;
    
    
end % end for iElem

        
