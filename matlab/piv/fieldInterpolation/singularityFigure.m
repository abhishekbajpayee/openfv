function [] = singularityFigure
% Draws a figure showing the position of singularities in the whittaker
% reconstruction

%   Copyright (c) 2007-2015  Thomas H. Clark
% Get a test array (gaussian blob)
sincTestArray = positionGaussian(zeros(48,48,48), [9 9 9], 1);
sincTestArray = sincTestArray/(max(sincTestArray(:)));

% Output window size of 41 chosen so that points pass through singularities
wSize = [41 41 41];

% Do the interpolation using two codes - one corrected for singularity, one
% not.
tic
[window windowCorrected] = singularWhitInterp3(sincTestArray, wSize);
toc

% Background grid to draw with
x2 = linspace(8,10,wSize(1));
[X2 Y2 Z2] = ndgrid(x2,x2,x2);

% Raise figure and subplots
raiseFigure('Singularity in Whittaker Reconstruction')
clf; 

% Plot the singularity zone
subplot(1,2,1)
hold on
ph1 = patch(isosurface(X2,Y2,Z2,window,      0.64));
set(ph1,'faceColor','b','faceAlpha',0.5,'edgeAlpha',0)
title('Singularities close to integer locations in x y and z')
grid on
view([-29, 24])
camlight right
lighting phong
axis equal
xlim([8 10])
ylim([8 10])
zlim([8 10])

subplot(1,2,2)
ph2 = patch(isosurface(X2,Y2,Z2,windowCorrected,      0.64));
set(ph2,'faceColor','g','edgeAlpha',0.2)
title('Singular behaviour corrected')
grid on
view([-29, 24])
camlight right
lighting phong
axis equal
xlim([8 10])
ylim([8 10])
zlim([8 10])




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
    
    
    
function [window windowOrig] = singularWhitInterp3(field, wSize)
% TODO: Document and expand for variable size and location window

% Number of samples (same in all directions for simplicity)
N = 7;

% Get window element indices in a list form
% [pMap] = windowCoords(c0, c1, c2, c3, c4, c5, c6, c7, wSize);
rVec = linspace(8,10,wSize(1));
cVec = linspace(8,10,wSize(1));
pVec = linspace(8,10,wSize(1));
[RVol CVol PVol] = ndgrid(rVec,cVec,pVec);
pMap = [RVol(:) CVol(:) PVol(:)];


% % Test that the windowcoords function produces the same...
% c0 = [8  8  8];
% c1 = [8  10 8];
% c2 = [10 10 8];
% c3 = [10  8 8];
% c4 = [8  8  10];
% c5 = [8  10 10];
% c6 = [10 10 10];
% c7 = [10  8 10];
% [pMap2] = windowCoords(c0, c1, c2, c3, c4, c5, c6, c7, wSize);
% isequal(pMap,pMap2)


% Get their integer positions
pMapRound = round(pMap);
offset = floor(N/2);

% And their local coord frame equivalents (location in local NxNxN frame)
tMap = pMap-pMapRound;


% Preallocate the window
window = zeros(wSize);
windowOrig = zeros(wSize);

% Get a hamming window
% w1 = hamming(N);
% Nope! It's crap - no filtering
w1 = ones(11,11,11);

% For each element
for iElem = 1:size(pMap,1);
    
    % Retrieve the 11^3 array surrounding the interpolation point: Amplitude at
    % sampling locations is contained in f_t
    row = pMapRound(iElem,1);
    col = pMapRound(iElem,2);
    dep = pMapRound(iElem,3);
    f_t = field(row-offset:row+offset,col-offset:col+offset,dep-offset:dep+offset);
    amplitude = 0;
    innerAmp = 0;
    
    for k = -offset:offset
        for j = -offset:offset
            for i = -offset:offset
                
                % NEW VERSION WITH EXPANDED SINE TERMS
                rMinusi = tMap(iElem,1)-i;
                cMinusj = tMap(iElem,2)-j;
                pMinusk = tMap(iElem,3)-k;
                m1powi  = (-1)^i;
                m1powj  = (-1)^j;
                m1powk  = (-1)^k;
                innerAmp = innerAmp + f_t(i+4,j+4,k+4)*m1powi*m1powj*m1powk/((pi^3)*rMinusi*cMinusj*pMinusk);
                
                
                
                % ORIGINAL VERSION
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
    % Original
	windowOrig(iElem) = amplitude;
    % New
    window(iElem) = innerAmp*sin(pi*tMap(iElem,1))*sin(pi*tMap(iElem,2))*sin(pi*tMap(iElem,3));
    
end % end for iElem

        


