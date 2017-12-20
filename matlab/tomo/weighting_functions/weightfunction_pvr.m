function [lookup_wts] = weightfunction_pvr(formstring,varargin)
%WEIGHTFUNCTION_PVR Returns a lookup table for pixel-voxel weighting
%
%   For use in tomographic reconstruction, a pixel line-of sight falls some
%   distance from the centre of any given voxel. The closer the l.o.s to the
%   centre of the voxel, the higher the 'weighting' between the pixel and the
%   voxel (i.e. the pixel contributes more to the intensity in that voxel in a 
%   MART reconstruction).
%
%   This function can be used where pixel to voxel ratios vary away from unity
%   (i.e. pixel size can be larger or smaller than a voxel). The lookup table is
%   given for pvr values 0.75:0.005:1.25
%
%   Here, unlike weightfunction.m, only a single functional relationship is 
%   used. It was shown using weightfunction (and found by authors such as 
%   Elsinga) that the circle-circle intersection (ref. 1) works best for 
%   practical tomography purposes, so there is no option to use gaussian 
%   weighting.
%
%   The fcnform argument is left active to facilitate future development and
%   investigation of the weighting function.
%
% Syntax:  
%       [lookup_wts] = weightfunction_pvr(fcnform)
%       [lookup_wts] = weightfunction_pvr(fcnform,'plot')
%
% Inputs:
%
%       fcnform     string      'circle'
%                               Currently, no other argument is acceptable (see
%                               above).
%
%       (optional)
%       'plot'                  Can be any variable or string (usually 'plot' is
%                               the most descriptive). Calling with this second
%                               argument causes a plot to be made of the
%                               functional variation.
%
% Outputs:
%		
%		lookup_wts  [227 x 101] real double
%                               A lookup table corresponding to the weightings
%                               for a range of distances. The first dimension
%                               (227) varies with distance for:
%                                       dist = sqrt(0:0.02:4.52)
%                               (the lookup range is based on the square of the
%                               distance for numerical efficiency in the MART
%                               code). The second dimension (100) varies with 
%                               pixel to voxel ratio 
% 
% Other files required:   	none
% Subfunctions:             none
% MAT-files required:       none
%
% References:
%   [1] http://mathworld.wolfram.com/Circle-CircleIntersection.html
%   [2] see weightfunction.m for version of this functioninvestigating gaussian
%       weighting functions (fixed unitary PVR).
%
% Future Improvements:
%       none foreseen
%
%
% Author:           T.H. Clark
% Work address:     Fluids Lab
%                   Cambridge University Engineering Department
%                   2 Trumpington Street
%                   Cambridge
%                   CB21PZ
% Email:            t.clark@cantab.net
% Website:          http://www.eng.cam.ac.uk/thc29
%
% Created:          unknown
% Documented:       17 March 2010
% Last revised:     17 March 2010

%   Copyright (c) 2007-2015  Thomas H. Clark
switch lower(formstring)
    
    case 'circle'
        
        % Currently, we assume that a voxel can be approximated by a circle whose area
        % is equal to the voxel size squared. Note this is an approximation which could
        % be improved upon if the angle of the line of sight was known.
        % In voxel units, the voxel size is 1, so A_circ = pi*R^2 = 1
        % It follows that R = 0.5642...
        r1sq = (1/pi);
        r1 = sqrt(r1sq);
        
        % By a similar argument, we can determine the effective radii corresponding to
        % the range of Pixel to Voxel ratios required. As voxel size (in voxel units) =
        % 1, pixel size (in voxel units) = pvr*r1.
        pixsize = (0.75:0.005:1.25)*r1;
        
        % We know from the explanation in the subroutine get_weights of mxmart_large.F90
        % that d^2 must vary between 0 and 4.52
        dsq = (0:0.02:4.52)';
        d = sqrt(dsq);
        
        % Grid the arrays:
        [d_array, r2_array]     = ndgrid(d,pixsize);
        [dsq_array, r2sq_array] = ndgrid(dsq,pixsize.^2); % Only for numerical efficiency
         
        % Mask problematic zones (negative, inf or NaN areas obtained from
        % numerical implementation of the mathematical formula).
        %   Where pixel circle entirely within voxel circle:
        r2mask = r2_array+d_array <= r1; % Area = area of pixel circle.
        % 	Where voxel circle entirely surrounded by pixel circle:
        r1mask = r1+d_array <= r2_array; % Area = area of voxel circle = 1
        %   Where voxel and pixel circles do not overlap
        dmask = d_array > (r1 + r2_array); % Area = 0
        %   Where we actually need to calculate the relationship:
        calcmask = ~(r2mask | r1mask | dmask);
        
        % Only calculate relationship for the points we need to
        d_vec = d_array(calcmask);
        r2_vec = r2_array(calcmask);
        dsq_vec = dsq_array(calcmask); % sqd terms precalculated for numerical efficiency
        r2sq_vec = r2sq_array(calcmask);
        
        % Predetermine the area
        area_array = pi.*r2sq_array;
        area_array(dmask) = 0;
        area_array(r1mask) = 1;
        %area_array(r2mask) = pixel area - already filled in
        
        
        % Using the extended formula for variable radii in the reference above:
        term1 = (r2sq_vec).*acos((dsq_vec + r2sq_vec - r1sq)./(2.*d_vec.*r2_vec));
        term2 = (r1sq).*acos((dsq_vec + r1sq - r2sq_vec)./(2*d_vec.*r1));
        term3 = (-d_vec+r2_vec+r1).*(d_vec+r2_vec-r1).*(d_vec-r2_vec+r1).*(d_vec+r1+r2_vec);
        area_vec = term1 + term2 - 0.5*sqrt(term3);
        
        % Mask the results back into the output array
        area_array(calcmask) = area_vec(:);
        
        lookup_wts = area_array;
        
        % Plot if required
        if nargin>1
            
            figure
            surf(dsq_array,r2_array./r1,area_array);
            xlabel('Distance^2 (voxels^2)')
            ylabel('Pixel to Voxel Ratio')
            zlabel('Weighting')
            
        end
        
    otherwise
        error('mxmart_large.m: weightfunction: Unrecognised fcnform string. Currently, ''circle'' is the only available option.')

end % End switch

end % Function weightfunction_pvr 


