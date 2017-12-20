function [lookup_wts] = weightfunction(formstring, varargin)
%WEIGHTFUNCTION Returns a lookup table for pixel-voxel weighting
%
%   For use in tomographic reconstruction, a pixel line-of sight falls some
%   distance from the centre of any given voxel. The closer the l.o.s to the
%   centre of the voxel, the higher the 'weighting' between the pixel and the
%   voxel (i.e. the pixel contributes more to the intensity in that voxel in a 
%   MART reconstruction).
%
%   This function is only valid where the assumption of unitary pixel to voxel
%   ratio holds (i.e. all pixels are approximately the same size as a single
%   voxel). Otherwise see ref. 3.
%
%   Here, two functional forms are available. 
%       'circle' models the weighting as a circle-circle intersection, using the
%          overlapping area as a weighting (see ref. 1).
%       'gaussian2' uses a gaussian form whose standard deviation is chosen such
%          that approx. 95% of the integrated area (2 std devs) lies at the same
%          point  where the circle-circle weighting diminishes to zero.
%       'gaussian3' is similar to gaussian2 except that 99.7% of the integrated
%          area (3 std devs) correspond with the circle-circle diminishing point
%
%   WARNING: The use of gaussians for weighting the reconstruction is not a good
%   idea! they can be helpful for understanding how the weighting affects the
%   reconstruction, but are useless for practical tomography!
%
% Syntax:  
%       [lookup_wts] = weightfunction(fcnform)
%       [lookup_wts] = weightfunction(fcnform,'plot')
%
% Inputs:
%
%       fcnform     string      'circle', 'gaussian2' or 'gaussian3'
%
%       (optional)
%       'plot'                  Can be any variable or string (usually 'plot' is
%                               the most descriptive). Calling weightfunction 
%                               with a second argument causes a plot to be made 
%                               of the functional variation. 
%
% Outputs:
%		
%		lookup_wts  [227 x 1]   Real double
%                               A lookup table corresponding to the weightings
%                               for a range of distances. The distances are
%                               sqrt(0:0.02:4.52) 
%                               (the lookup range is based on the square of the
%                               distance for numerical efficiency in the MART
%                               code).
% 
% Other files required:   	none
% Subfunctions:             none
% MAT-files required:       none
%
% References:
%   [1] http://mathworld.wolfram.com/Circle-CircleIntersection.html
%   [2] http://en.wikipedia.org/wiki/Normal_distribution
%   [3] see weightfunction_pvr for extension of this function to allow for
%       varying pixel to voxel ratios.
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


switch lower(formstring(1:end-1))
    
    case 'circl'
        
        % Currently, we assume that a voxel can be approximated by a circle whose area
        % is equal to the voxel size squared. Note this is an approximation which could
        % be improved upon if the angle of the line of sight was known.
        % In voxel units, the voxel size is 1, so A_circ = pi*R^2 = 1
        % It follows that R = 0.5642...
        rsqd = (1/pi);
        r = sqrt(rsqd);
        
        % We know from the explanation in the subroutine get_weights of mxmart_large.F90
        % that d^2 must vary between 0 and 4.52
        dsq = 0:0.02:4.52;
        d = sqrt(dsq');
        masker = d>(2*r);
        
        % Using the formula from wolfram mathworld (vectorially):
        lookup_wts = 2*rsqd*acos(d/(2*r)) - 0.5*d.*(((4*rsqd) - (d.^2)).^0.5);
        
        % Prevent complex numbers occurring (-ve areas)
        lookup_wts(masker) = 0;
        
        % Plot (if required) a figure containing the weight relationship.
        if nargin > 1
            figure()
            plot(d,lookup_wts);
            xlabel('distance in voxels');
            ylabel('weighting');
            title('Circular Intersection Pixel-Voxel Weighting Function (pvr = 1)')
        end
        
        
    case 'gaussian' 
        
        % For a pixel of characteristic dimension 1 voxel, where both pixel and
        % voxel are modelled as circles, the point of first intersection lies at
        % d = 2r = 2*sqrt(1/pi)
        dcrit = 2*sqrt(1/pi);
        
        % Here, this distance is equal to twice or three times the standard 
        % deviation:
        sigma = dcrit./str2double(formstring(end));
        
        % Distances required are:
        d = sqrt(0:0.02:4.52)';
        
        % Gaussian centred around a mean d = 0
        lookup_wts = exp(-0.5*(d./sigma).^2)./(sqrt(2*pi).*sigma);
        
        % Normalise the distribution to give a weighting of 1 at d = 0:
        lookup_wts = lookup_wts./max(lookup_wts);
        
        % Plot (if required) a figure containing the weight relationship.
        if nargin > 1
            figure()
            plot(d,lookup_wts);
            xlabel('Distance (voxels)');
            ylabel('Weighting');
            title(['Gaussian' formstring(end) ' Pixel-Voxel Weighting Function (pvr = 1)'])
        end
        
        
    otherwise
        
        error('mxmart_large.m: weightfunction: Unrecognised fcnform string. try ''circle'', ''gaussian2'' or ''gaussian3''.')
end


end
