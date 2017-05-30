function [ur uc up snr] = cte3d(fieldA, fieldB, fieldC, fieldD, c0, c1, c2, c3, c4, c5, c6, c7, wSize, iwMaxDisp, fetchType, parFlag)
%CTE3D Matlab wrapper for the mex function mex_cte3d

%   Copyright (c) 2007-2015  Thomas H. Clark

% Make checks that the cardinal interpolation for the windows will not
% cause out of bounds errors
fieldSize = size(fieldA);


% Check fields are not empty
if isempty(fieldA) || isempty(fieldB) || isempty(fieldC) || isempty(fieldD) 
    error('MATLAB:TomoPIVToolbox:IncorrectSize','Field arrays A, B, C and D must not be empty')
end
% Check fields are the right size
if ~isequal(fieldSize, size(fieldB)) || ~isequal(fieldSize, size(fieldC)) || ~isequal(fieldSize, size(fieldD)) 
    error('MATLAB:TomoPIVToolbox:IncorrectSize','Field arrays A, B, C and D are different sizes')
end

% Check that c0 et are all the same size
cSize = size(c0);
if ~isequal(cSize, size(c1), size(c2), size(c3), size(c4), size(c5), size(c6), size(c7))
    error('MATLAB:TomoPIVToolbox:IncorrectSize','Corner location arrays are different sizes')
end

% Check that arrays are of minimum size at least
if any(fieldSize <= 7)
    error('MATLAB:TomoPIVToolbox:IncorrectSize','Field array must be >7 elements in every dimension')
end
if (cSize(2) ~= 3)
    error('MATLAB:TomoPIVToolbox:IncorrectSize','Window corner location arrays must have 3 columns')
end
if (cSize(1) < 2)
    error('MATLAB:TomoPIVToolbox:IncorrectSize','Window corner location arrays must have at least 2 rows')
end
if (rem(cSize(1), 2) ~= 0)
    error('MATLAB:TomoPIVToolbox:IncorrectSize','Window corner location arrays have 2N rows where N is the number of cross correlations')
end

% Check that corner arrays won't give out of bounds
if any(round(c0(:)) <= 3) || any(round(c1(:)) <= 3) || any(round(c2(:)) <= 3) || any(round(c3(:)) <= 3) || any(round(c4(:)) <= 3) || any(round(c5(:)) <= 3) || any(round(c6(:)) <= 3) || any(round(c7(:)) <= 3)
    error('MATLAB:TomoPIVToolbox:IncorrectSize','Window corner location arrays are too close to the field boundary for reliable cardinal interpolation. Try setting the edgeCut option higher (>=3)')
end
sz = size(fieldA)-3;
sz = repmat([sz(2) sz(1) sz(3)], [size(c0,1), 1]);
if any(any(round(c0) > sz) | any(round(c1) > sz) | any(round(c2) > sz) | any(round(c3) > sz) | any(round(c4) > sz) | any(round(c5) > sz) | any(round(c6) > sz) | any(round(c7) > sz))
    error('MATLAB:TomoPIVToolbox:IncorrectSize','Window corner location arrays are too close to the field boundary for reliable cardinal interpolation. Try setting the edgeCut option higher (>=3)')
end
% Check that window size is valid
if ~isequal(size(wSize),[1 1])
    error('MATLAB:TomoPIVToolbox:IncorrectSize','Window size parameter must be a scalar')
end
if (wSize < 2)
    error('MATLAB:TomoPIVToolbox:IncorrectSize','Window size must be >= 2^3 voxels')
elseif (wSize > 64)
    warning('MATLAB:TomoPIVToolbox:IncorrectSize','Window size is > 64^3 voxels which may cause out of memory errors and extremely slow performance')
end

% Check that iwMaxDisp paramter is scalar and in correct bounds
if ~isequal(size(iwMaxDisp),[1 1])
    error('MATLAB:TomoPIVToolbox:IncorrectSize','iwMaxDisp parameter must be a scalar')
end
if (iwMaxDisp < 1) || (iwMaxDisp > 100)
    error('MATLAB:TomoPIVToolbox:IncorrectSize','iwMaxDisp parameter is expressed as a percentage: (varying between 1 and 100)')
end

% Check that fetchtype is sensible
if (fetchType ~= 1) && (fetchType ~= 2) && (fetchType ~= 3) && (fetchType ~= 4)
    error('MATLAB:TomoPIVToolbox:IncorrectType','Window fetch type must be 1 (direct), 2 (linear), 3 (5^3 cardinal) or 4 (7^3 cardinal)')
end

% Cast to appropriate typing (fMexPIV not checking yet so will cause 
% segVs if not cast correctly)
fieldA = single(fieldA);
fieldB = single(fieldB);
fieldC = single(fieldC);
fieldD = single(fieldD);
c0 = single(c0);
c1 = single(c1);
c2 = single(c2);
c3 = single(c3);
c4 = single(c4);
c5 = single(c5);
c6 = single(c6);
c7 = single(c7);
wSize = int32(wSize);
iwMaxDisp = single(iwMaxDisp);
fetchType = int32(fetchType);


% Save temporary file to debug the window fetching
% save(['windowFetchDebugData' datestr(now) '.mat'],'c0','c1','c2','c3','c4','c5','c6','c7','wSize','iwMaxDisp','fetchType')

% OpenMP places a _massive_ burden on the stack. Temporary arrays can be
% quite large. So...
!ulimit -s unlimited

% Make the direct call to mex_cte3d
[ur uc up snr] = mex_cte3d(fieldA, fieldB, fieldC, fieldD, c0, c1, c2, c3, c4, c5, c6, c7, wSize, iwMaxDisp, fetchType);

% Issue a clear mex to remove any persistents left by the mex file
clear mex
    




% Cast back to double in order to preserve functionality of MATLAB's
% downstream routines such as TriScatteredInterp.
ur  = double(ur);
uc  = double(uc);
up  = double(up);
snr = double(snr);
