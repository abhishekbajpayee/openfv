function [window windowOrig] = whitInterp3(field, wSize)
% TODO: Document and expand for variable size and location window

%   Copyright (c) 2007-2015  Thomas H. Clark
% Number of samples (same in all directions for simplicity)
N = 7;

% Get window element indices in a list form
% [pMap] = windowCoords(c0, c1, c2, c3, c4, c5, c6, c7, wSize);
rVec = linspace(7.95,10.05,wSize(1));
cVec = linspace(7.95,10.05,wSize(1));
pVec = linspace(7.95,10.05,wSize(1));
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

% Correct to prevent singular behaviour
tMap(abs(tMap) <= 10*eps('single')) = 10*eps('single');

% Preallocate the window
window = zeros(wSize);
windowOrig = zeros(wSize);

% Get a hamming window
w1 = hamming(N);
% Nope! It's crap - no filtering
w1 = ones(size(w1));

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
                rMult = m1powi/rMinusi;
                cMult = m1powj/cMinusj;
                pMult = m1powk/pMinusk;
                amp =  f_t(i+4,j+4,k+4)*rMult*cMult*pMult;
                innerAmp = innerAmp + amp;
                
                
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
    
    
    rSin = sin(pi*tMap(iElem,1));
    cSin = sin(pi*tMap(iElem,2));
    pSin = sin(pi*tMap(iElem,3));
    window(iElem) = innerAmp*rSin*cSin*pSin/(pi^3);
    
end % end for iElem

        

end % end function whitInterp3






function [pMap] = windowCoords(p0, p1, p2, p3, p4, p5, p6, p7, wSize)
% Returns [r,c,p] window element coordinates in list form, created from the
% corner coordinates and window size. Note the order of the for loops returns it
% as a list reshaped from row,col,dep array

% 1 based corner indices in = 1 based positons out.
% 0 based corner indices in = 0 based positions out.

% These might as well all be the same - never going to use windows other than
% cubes.
szM = wSize(1);
szN = wSize(2);
szP = wSize(3);

% pMap contains zero based indices into the voxels array of the elements in the 
pMap = zeros(prod(wSize),3);
pCtr = 1;

for p = 1:wSize(3)
    for c = 1:wSize(2)
        for r = 1:wSize(1)

            % Determine the fractional position in the x,y,z directions within 
            % the window (float, varies between 0 and 1).
            f1 = (c-1) / (szM-1);
            f2 = (r-1) / (szN-1);
            f3 = (p-1) / (szP-1);

            % Using the fractional positions, we map the output position r,c,p 
            % to the row, column and page in the texture.
            % 
            % This requires knowledge of the ORDER in which corner indices are
            % stored in the wCorner vectors. Order is as follows:
            %          0       low row, low col, low page
            %          1       high row, low col, low page
            %          2       high row, high col, low page
            %          3       low row, high col, low page
            %          4       low row, low col, high page
            %          5       high row, low col, high page
            %          6       high row, high col, high page
            %          7       low row, high col, high page
            %          8       repeat for next window in same order
            % 
            % The photo 'windowOrdering.jpg' shows the diagram I've drawn to
            % illustrate this (in the source code folder)
            % 
            % Note that the output position in the texture pMap already 
            % consists of C,R,P indices (into the texture) and does not
            % need to be scaled or normalised by the texture dimensions
            p31 = f3*(p4 - p0) + p0;
            p32 = f3*(p5 - p1) + p1;
            p33 = f3*(p7 - p3) + p3;
            p34 = f3*(p6 - p2) + p2;

            p41  = f1*(p33 - p31) + p31;
            p42  = f1*(p34 - p32) + p32;
            pMap(pCtr,:) = f2*(p42 - p41) + p41;


            % Increment the counter
            pCtr = pCtr + 1;
            
        end
    end
end

% adjust to row,col,page triplets instead of col,row,page
pMap(:,1:2) = fliplr(pMap(:,1:2));



end % end function windowCoords
