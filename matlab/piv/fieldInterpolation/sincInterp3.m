function [window] = sincInterp3(field, c0, c1, c2, c3, c4, c5, c6, c7, wSize)
%SINCINTERP3 Interpolates regions of a 3d intensity field to regular grids
%
% Syntax:
%
%        [window] = sincInterp3(field, c0, c1, c2, c3, c4, c5, c6, c7, wSize)
%
% Inputs:
%
% Outputs:
%
% Examples:
%
% References:
%
%   [1] Lourenco L. and Krothapalli A. (1995) On the accuracy of velocity and
%       vorticity measurements with PIV. Experiments in Fluids 18 pp. 421-428
%
%   [2] Chen J. and Katz J. (2005) Elimination of the peak-locking error in PIV
%       analysis using the correlation mapping method. 
%       Meas. Sci. Technol. 16, 1605-1618
%
%   [3] Stearns S.D. and Hush D. (1990) Digital Signal Analysis. Second Edition.
%       Prentice-Hall pp. 75-84
%
% Future Improvements:
% Other m-files required:   none
% Subfunctions:             none
% Nested functions:         none
% MAT-files required:       none
%
%
% Author:                   T.H. Clark
% Work address:             Fluids Lab
%                           Cambridge University Engineering Department
%                           2 Trumpington Street
%                           Cambridge
%                           CB21PZ
% Email:                    t.clark@cantab.net
% Website:                  http://cambridge.academia.edu/ThomasClark/
%
% Revision History:         01 May 2011         Created
%                           02 July 2011        Restructured and finished
%                                               writing

%   Copyright (c) 2007-2015  Thomas H. Clark
% Not working. I've recoded the same functionality into whitInterp3 which works
% correctly.
% After extensive testing and debugging, I still couldn't get this to work
% properly. At the end of this file, commented out, is the version of this file
% which I had AFTER all that testing and debugging (for my future use).
error('sincInterp3: Not working properly. Use whitInterp3 instead')

% Get window element indices in a list form
[pMap] = windowCoords(c0, c1, c2, c3, c4, c5, c6, c7, wSize);

% Get their integer equivalents
pMapRound = round(pMap);

% And their local coord frame equivalents (location in local 5x5x5 frame)
tMap = pMap-pMapRound+2;

% Period; same in all 3 directions, for simplicity. This is the same as
% assuming voxels in the reconstruction are cubes. The period is = 1
T = 1;

% Number of samples (same in all directions for simplicity)
N = 5;

% Integer values of sample location, referred to the local 5x5 grid
% (zero-based).
m = 0:(N-1);
n = 0:(N-1);
p = 0:(N-1);
m = m(:);   % m is a column vector representing rows in the input matrix
n = n(:)';  % n is row vector representing columns in the input matrix
p = reshape(p, [1 1 numel(p)]);	% p is a depth vector representing pages in the input matrix

% Preallocate the iwndow
window = zeros(wSize);

% For each element
for iElem = 1:size(pMap,1);
    
    % Retrieve the 11^3 array surrounding the interpolation point: Amplitude at
    % sampling locations is contained in f_t
    row = pMapRound(iElem,2);
    col = pMapRound(iElem,1);
    dep = pMapRound(iElem,3);
    f_t = field(row-2:row+2,col-2:col+2,dep-2:dep+2);
        
    % Perform the interpolation at this location. NB this is also structured so
    % we can use a different option of breaking the field up into buckets.
    [amplitude] = whittakerInterpolate(tMap(iElem,:), f_t, T, N, m, n, p);
    
    % The pMap is produced in column, row order... so we need a smart way of
    % indexing back into the window... may need to transpose
    window(iElem) = amplitude;
    
    
end % end for iElem

        
end % end main function      

    
function [amplitude] = whittakerInterpolate(tPoints, peakArray, T, nSamples, m, n, p)

% tPoints is a matrix containing locations at which to interpolate values of the
% function. It is of size [n x 3] where each row contains an x,y,z location
% within the bounds of m, n and p respectively.

% NEED TO VECTORISE! (Or would be perfect for embedding)
nPoints = size(tPoints,1);
amplitude = zeros([nPoints 1]);

for iPoint = 1:nPoints
    
    % Current tPoint
    tPoint = tPoints(iPoint,:);
        
    % Value of sin(pi*t1/T)*sin(pi*t2/T)*sin(pi*t3/T)
    %sinValue = sin( (pi*tPoint(1)/T) ) .* sin( (pi*tPoint(2)/T) ) .* sin( (pi*tPoint(3)/T) );
    
    % Values of t-mT, t-nT and t-pT
    tMinusmT = (tPoint(1) - m.*T); % column vector
    tMinusnT = (tPoint(2) - n.*T); % row vector
    tMinuspT = (tPoint(3) - p.*T); % depth vector
    
    % Terms in the summation. The if statement deals with singularity occurring
    % when t-XT = 0, i.e. the interpolation point lies on a straight line
    % connecting the grid points. Note also the nCub term - where we deal with a
    % singularity, we effectively revert to 2D or 1D cardinal interpolation of
    % the nonsingular dimensions so the exponent of (pi/T)^N reduces to reflect
    % this. In the canonical case where the value is at a grid point, the
    % interpolation reduces to 0D and simply results in the value of the input
    % array at that grid point.
    nCub = 3;
    if tMinusmT(3) == 0
        mTerm = zeros([5 1]);            % column vector
        mTerm(3) = 1;
        mSinTerm = 1;
        nCub = nCub-1;
    else
        mTerm = ((-1).^m)./tMinusmT;	% column vector
        mSinTerm = sin( (pi*tPoint(1)/T) );
    end
    if tMinusnT(3) == 0
        nTerm = zeros([1 5]);            % row vector
        nTerm(3) = 1;
        nSinTerm = 1;
        nCub = nCub-1;
    else
        nTerm = ((-1).^n)./tMinusnT;	% row vector
        nSinTerm = sin( (pi*tPoint(2)/T) );
    end
    if tMinuspT(3) == 0
        pTerm = zeros([1 1 5]);          % depth vector
        pTerm(3) = 1;
        pSinTerm = 1;
        nCub = nCub-1;
    else
        pTerm = ((-1).^p)./tMinuspT;	% depth vector
        pSinTerm = sin( (pi*tPoint(3)/T) );
    end
    
    % Value of the external sine term (sin(pi*t1/T)*sin(pi*t2/T)*sin(pi*t3/T))
    sinValue = mSinTerm*nSinTerm*pSinTerm;   
    
    % Expand matrices along singleton dimension and multiply for inner term
    mTerm = repmat(mTerm, [1 nSamples nSamples]);
    nTerm = repmat(nTerm, [nSamples 1 nSamples]);
    pTerm = repmat(pTerm, [nSamples nSamples 1]);
    innerTerm = peakArray.*mTerm.*nTerm.*pTerm;
    
    % Determine the value by summation
    amplitude(iPoint) = sinValue*sum(sum(sum(innerTerm,3),2),1)./((pi/T)^nCub);
    
    % None of that works if the point is actually one of the locations in the
    % input grid. In that event, stitch in the known value:
%     if isequal(tPoint, round(tPoint))
%         amplitude(iPoint) = peakArray(tPoint(1)+1, tPoint(2)+1, tPoint(3)+1);
%     end
    
end % end for iPoint

end % End function whittakerInterpolate






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
    for c = 1:wSize(1)
        for r = 1:wSize(2)

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
% pMap(:,1:2) = fliplr(pMap(:,1:2));



end % end function windowCoords

            
            





% DEBUGGED BUT STILL NOT WORKING
% 
% 
% function [window] = sincInterp3(field, c0, c1, c2, c3, c4, c5, c6, c7, wSize)
% %SINCINTERP3 Interpolates regions of a 3d intensity field to regular grids
% %
% % Syntax:
% %
% %        [window] = sincInterp3(field, c0, c1, c2, c3, c4, c5, c6, c7, wSize)
% %
% % Inputs:
% %
% % Outputs:
% %
% % Examples:
% %
% % References:
% %
% %   [1] Lourenco L. and Krothapalli A. (1995) On the accuracy of velocity and
% %       vorticity measurements with PIV. Experiments in Fluids 18 pp. 421-428
% %
% %   [2] Chen J. and Katz J. (2005) Elimination of the peak-locking error in PIV
% %       analysis using the correlation mapping method. 
% %       Meas. Sci. Technol. 16, 1605-1618
% %
% %   [3] Stearns S.D. and Hush D. (1990) Digital Signal Analysis. Second Edition.
% %       Prentice-Hall pp. 75-84
% %
% % Future Improvements:
% % Other m-files required:   none
% % Subfunctions:             none
% % Nested functions:         none
% % MAT-files required:       none
% %
% %
% % Author:                   T.H. Clark
% % Work address:             Fluids Lab
% %                           Cambridge University Engineering Department
% %                           2 Trumpington Street
% %                           Cambridge
% %                           CB21PZ
% % Email:                    t.clark@cantab.net
% % Website:                  http://cambridge.academia.edu/ThomasClark/
% %
% % Revision History:         01 May 2011         Created
% %                           02 July 2011        Restructured and finished
% %                                               writing
% 
% 
% % Period; same in all 3 directions, for simplicity. This is the same as
% % assuming voxels in the reconstruction are cubes. The period is = 1
% T = 1;
% 
% % Number of samples (same in all directions for simplicity)
% N = 15;
% 
% % Get window element indices in a list form
% % [pMap] = windowCoords(c0, c1, c2, c3, c4, c5, c6, c7, wSize);
% rVec = linspace(8,10,wSize(1));
% cVec = linspace(8,10,wSize(1));
% pVec = linspace(8,10,wSize(1));
% [RVol CVol PVol] = ndgrid(rVec,cVec,pVec);
% pMap = [RVol(:) CVol(:) PVol(:)];
% 
% % Get their integer positions
% pMapRound = floor(pMap);
% 
% % And their local coord frame equivalents (location in local 5x5x5 frame)
% offset = floor(N/2);
% tMap = pMap-pMapRound+offset;
% 
% 
% % 3D Hamming window by the outer product method
% i = 0:(N-1);
% w1 = -0.46*cos(2*pi*i/(N-1))+0.54;
% b1 = -0.5*cos(2*pi*i/(N-1)) + 0.08*cos(4*pi*i/(N-1)) +0.42;
% % w1 = hamming(N);
% % figure()
% % plot(i,w1);hold on; plot(i,b1,'g-')
% w1 = b1(:);
% w2 = b1(:)';
% w3 = reshape(b1,[1 1 numel(b1)]);
% % w1 = hamming(N);
% [rVol, cVol, pVol] = ndgrid(-offset:offset);
% r = sqrt(rVol.^2 + cVol.^2 + pVol.^2);
% w = zeros(size(r));
% w(r<=offset) = interp1(-offset:offset,w1,r(r<=offset),'pchip');
% w = ones(size(r));
% 
% 
% % Integer values of sample location, referred to the local 5x5 grid
% % (zero-based).
% m = 0:(N-1);
% n = 0:(N-1);
% p = 0:(N-1);
% % m = -offset:offset;
% % n=m;p=m;
% m = m(:);   % m is a column vector representing rows in the input matrix
% n = n(:)';  % n is row vector representing columns in the input matrix
% p = reshape(p, [1 1 numel(p)]);	% p is a depth vector representing pages in the input matrix
% 
% % Preallocate the iwndow
% window = zeros(wSize);
% 
% % For each element
% for iElem = 1:size(pMap,1);
%     
%     % Retrieve the 11^3 array surrounding the interpolation point: Amplitude at
%     % sampling locations is contained in f_t
%     row = pMapRound(iElem,1);
%     col = pMapRound(iElem,2);
%     dep = pMapRound(iElem,3);
%     f_t = field(row-offset:row+offset,col-offset:col+offset,dep-offset:dep+offset);
%         
%     % Perform the interpolation at this location. NB this is also structured so
%     % we can use a different option of breaking the field up into buckets.
%     [amplitude] = whittakerInterpolate(tMap(iElem,:), f_t, T, N, m, n, p, w1, w2, w3, w);
%     
%     window(iElem) = amplitude;
%     
%     
% end % end for iElem
% 
%         
% end % end main function      
% 
% % function [amplitude] =scaranoInterpolate(tPoint, peakArray, T, nSamples, m, n, p)
% % 
% % for i = 1:7
% % 
% % end
%     
% function [amplitude] = whittakerInterpolate(tPoints, peakArray, T, nSamples, m, n, p, w1, w2, w3, w)
% 
% % tPoints is a matrix containing locations at which to interpolate values of the
% % function. It is of size [n x 3] where each row contains an x,y,z location
% % within the bounds of m, n and p respectively.
% 
% % NEED TO VECTORISE! (Or would be perfect for embedding)
% nPoints = size(tPoints,1);
% amplitude = zeros([nPoints 1]);
% 
% for iPoint = 1:nPoints
%     
%     % Current tPoint
%     tPoint = tPoints(iPoint,:);
%         
%     % Value of sin(pi*t1/T)*sin(pi*t2/T)*sin(pi*t3/T)
%     %sinValue = sin( (pi*tPoint(1)/T) ) .* sin( (pi*tPoint(2)/T) ) .* sin( (pi*tPoint(3)/T) );
%     
%     % Values of t-mT, t-nT and t-pT
%     tMinusmT = (tPoint(1) - m.*T); % column vector
%     tMinusnT = (tPoint(2) - n.*T); % row vector
%     tMinuspT = (tPoint(3) - p.*T); % depth vector
%     
%     % Terms in the summation. The if statement deals with singularity occurring
%     % when t-XT = 0, i.e. the interpolation point lies on a straight line
%     % connecting the grid points. Note also the nCub term - where we deal with a
%     % singularity, we effectively revert to 2D or 1D cardinal interpolation of
%     % the nonsingular dimensions so the exponent of (pi/T)^N reduces to reflect
%     % this. In the canonical case where the value is at a grid point, the
%     % interpolation reduces to 0D and simply results in the value of the input
%     % array at that grid point.
%     nCub = 3;
%     if tMinusmT(3) == 0
%         mTerm = zeros([nSamples 1]);            % column vector
%         mTerm(3) = 1;
%         mSinTerm = 1;
%         nCub = nCub-1;
%     else
%         mTerm = ((-1).^(m+1))./tMinusmT;	% column vector
%         mSinTerm = sin( (pi*tPoint(1)/T) );
% %          m2 = w1.*sinc( (pi*tMinusmT/T) );
%     end
%     if tMinusnT(3) == 0
%         nTerm = zeros([1 nSamples]);            % row vector
%         nTerm(3) = 1;
%         nSinTerm = 1;
%         nCub = nCub-1;
%     else
%         nTerm = ((-1).^(n+1))./tMinusnT;	% row vector
%         nSinTerm = sin( (pi*tPoint(2)/T) );
% %          n2 = w2.*sinc( (pi*tMinusnT/T) );
%     end
%     if tMinuspT(3) == 0
%         pTerm = zeros([1 1 nSamples]);          % depth vector
%         pTerm(3) = 1;
%         pSinTerm = 1;
%         nCub = nCub-1;
%     else
%         pTerm = ((-1).^(p+1))./tMinuspT;	% depth vector
%         pSinTerm = sin( (pi*tPoint(3)/T) );
% %         p2 = w3.*sinc( (pi*tMinuspT/T) );
%     end
%     
%     % Value of the external sine term (sin(pi*t1/T)*sin(pi*t2/T)*sin(pi*t3/T))
%     sinValue = mSinTerm*nSinTerm*pSinTerm;   
%     
%     % Expand matrices along singleton dimension and multiply for inner term
%     mTerm = repmat(mTerm, [1 nSamples nSamples]);
%     nTerm = repmat(nTerm, [nSamples 1 nSamples]);
%     pTerm = repmat(pTerm, [nSamples nSamples 1]);
% %     mTerm = repmat(m2, [1 nSamples nSamples]);
% %     nTerm = repmat(n2, [nSamples 1 nSamples]);
% %     pTerm = repmat(p2, [nSamples nSamples 1]);
%     innerTerm = w.*peakArray.*mTerm.*nTerm.*pTerm;
%     
%     % Determine the value by summation
%     amplitude(iPoint) = sinValue*sum(sum(sum(innerTerm,3),2),1)./((pi/T)^nCub);
% %     amplitude(iPoint) = sum(sum(sum(innerTerm,3),2),1);
% %     
%     % None of that works if the point is actually one of the locations in the
%     % input grid. In that event, stitch in the known value:
% %     if isequal(tPoint, round(tPoint))
% %         amplitude(iPoint) = peakArray(tPoint(1)+1, tPoint(2)+1, tPoint(3)+1);
% %     end
%     
% end % end for iPoint
% 
% end % End function whittakerInterpolate
% 
% 
% 
% 
% 
% 
% function [pMap] = windowCoords(p0, p1, p2, p3, p4, p5, p6, p7, wSize)
% % Returns [r,c,p] window element coordinates in list form, created from the
% % corner coordinates and window size. Note the order of the for loops returns it
% % as a list reshaped from row,col,dep array
% 
% 
% % 1 based corner indices in = 1 based positons out.
% % 0 based corner indices in = 0 based positions out.
% 
% % These might as well all be the same - never going to use windows other than
% % cubes.
% szM = wSize(1);
% szN = wSize(2);
% szP = wSize(3);
% 
% % pMap contains zero based indices into the voxels array of the elements in the 
% pMap = zeros(prod(wSize),3);
% pCtr = 1;
% 
% for p = 1:wSize(3)
%     for c = 1:wSize(2)
%         for r = 1:wSize(1)
% 
%             % Determine the fractional position in the x,y,z directions within 
%             % the window (float, varies between 0 and 1).
%             f1 = (c-1) / (szM-1);
%             f2 = (r-1) / (szN-1);
%             f3 = (p-1) / (szP-1);
% 
%             % Using the fractional positions, we map the output position r,c,p 
%             % to the row, column and page in the texture.
%             % 
%             % This requires knowledge of the ORDER in which corner indices are
%             % stored in the wCorner vectors. Order is as follows:
%             %          0       low row, low col, low page
%             %          1       high row, low col, low page
%             %          2       high row, high col, low page
%             %          3       low row, high col, low page
%             %          4       low row, low col, high page
%             %          5       high row, low col, high page
%             %          6       high row, high col, high page
%             %          7       low row, high col, high page
%             %          8       repeat for next window in same order
%             % 
%             % The photo 'windowOrdering.jpg' shows the diagram I've drawn to
%             % illustrate this (in the source code folder)
%             % 
%             % Note that the output position in the texture pMap already 
%             % consists of C,R,P indices (into the texture) and does not
%             % need to be scaled or normalised by the texture dimensions
%             p31 = f3*(p4 - p0) + p0;
%             p32 = f3*(p5 - p1) + p1;
%             p33 = f3*(p7 - p3) + p3;
%             p34 = f3*(p6 - p2) + p2;
% 
%             p41  = f1*(p33 - p31) + p31;
%             p42  = f1*(p34 - p32) + p32;
%             pMap(pCtr,:) = f2*(p42 - p41) + p41;
% 
% 
%             % Increment the counter
%             pCtr = pCtr + 1;
%             
%         end
%     end
% end
% % MODIFICATION FROM FORTRAN - Added this
% % adjust to row,col,page triplets instead of col,row,page
% pMap(:,1:2) = fliplr(pMap(:,1:2));
% 
% 
% 
% end % end function windowCoords
% 
%             
%             
%             
%             
%             
%             
