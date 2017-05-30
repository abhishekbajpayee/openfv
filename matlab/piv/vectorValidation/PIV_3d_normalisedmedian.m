function [med_test, nmr] = PIV_3d_normalisedmedian(ux, uy, uz, eps_PIV_noise, eps_threshold)
%PIV_3D_NORMALISEDMEDIAN Vector field validation using normalised local median.
%   Approach as per ref [1], p.185.
%
% Syntax:  
%       [med_test, r_med] = PIV_3d_normalisedmedian(ux, uy, uz, eps_PIV_noise, eps_threshold)
%
% Inputs:
%       ux      [ny, nx, nz]    3D array containing x-components of velocity
%                               field.
%
%       uy      [ny, nx, nz]    3D array containing y-components of velocity
%                               field.   
%
%       uy      [ny, nx, nz]    3D array containing z-components of velocity
%                               field.
%
%       eps_PIV_noise           scalar, typically 0.1-0.2 
%                               Mean noise level of the PIV data. This can be
%                               determined by plotting a histogram of intensity
%                               in the reconstructed volumes, and observing the
%                               amount of low-level noise.
%
%      
%       eps_threshold           No idea what this value should take. 
%                               Threshold of median test acceptability, denoted
%                               as epsilon_thresh, p.185, section 6.1.5, ref[1].
%
% Outputs:
%
%       med_test    [ny nx nz]  logical array 
%                               The same size as input velocity arrays. True
%                               indicates the the vector is invalid, according
%                               to the normalised median test criterion.
%
%       nmr         [ny nx nz]  Contains the values of normalised median used to
%                               perform the validity test. Plotting a histogram
%                               of these values (highlighting invalid vectors)
%                               will help the user to ascertain appropriate
%                               eps_threshold value for stringency (also see
%                               p.186 ref [1]).
%
% Examples:
%       see PIV_3d_vectorcheck.m for example usage in context.
%
% References:
%   [1] Raffal M. Willert C. Wereley S. and Kompenhans J. 
%       'Particle Image Velocimetry (A Practical Guide)' 
%       2nd Ed., Springer,  ISBN 978-3-540-72307-3
%
% Future Improvements:      none
% Other m-files required:   none
% Subfunctions:             none
% Nested functions:         none
% MAT-files required:       none
%
%
% Author:           T.H. Clark
% Work address:     Fluids Lab
%                   Cambridge University Engineering Department
%                   2 Trumpington Street
%                   Cambridge
%                   CB21PZ
% Email:            t.clark@cantab.net
% Website:          http://cambridge.academia.edu/ThomasClark/
%
% Created:          20 October 2009
% Last revised:     28 October 2009

%   Copyright (c) 2007-2015  Thomas H. Clark
% Include the RTW embedded MATLAB pragma
%#eml


% Given a velocity field ux, uy, uz (currently assumed to be spaced the same
% amount in each direction) compute the normalised median velocity at each
% point.

% At each point, calculate the median of the 26 surrounding points:
%   NB do it first based on velocity magnitude then on individual components
u_mag = sqrt(ux.^2 + uy.^2 + uz.^2);

% NB we return the residuals of the magnitude one only
[med_test_umag,nmr] = med26(u_mag, eps_PIV_noise, eps_threshold);
[med_test_ux]   = med26(ux, eps_PIV_noise, eps_threshold);
[med_test_uy]   = med26(uy, eps_PIV_noise, eps_threshold);
[med_test_uz]   = med26(uz, eps_PIV_noise, eps_threshold);

% Final stringent median test:
med_test = med_test_umag | med_test_ux | med_test_uy | med_test_uz;


end % end main function
function [med_test,nmr] = med26(u_mag, eps_PIV_noise, eps_threshold)
    % THIS NEEDS DOCUMENTING PROPERLY (variables etc must be renamed properly
    % too)
%   Create size templates
u_size = size(u_mag);
if numel(u_size) ==2
    % Then we have a single layer of windows in the Z direction - but, the size
    % array comes out as a two element vector (e.g. [13 13] instead of [13 13
    % 1]). Compensate for this:
    u_size = [u_size 1];
end
blank_size = u_size + [2 2 2];

%   Create 26 blank arrays, containing NaNs (allowing us to easily mask borders)
arr01 = NaN(blank_size);
arr02 = arr01;
arr03 = arr01;
arr04 = arr01;
arr05 = arr01;
arr06 = arr01;
arr07 = arr01;
arr08 = arr01;
arr09 = arr01;
arr10 = arr01;
arr11 = arr01;
arr12 = arr01;
arr13 = arr01;
arr14 = arr01;
arr15 = arr01;
arr16 = arr01;
arr17 = arr01;
arr18 = arr01;
arr19 = arr01;
arr20 = arr01;
arr21 = arr01;
arr22 = arr01;
arr23 = arr01;
arr24 = arr01;
arr25 = arr01;
arr26 = arr01;

% Put the velocity distribution into these arrays at a different starting
% indices
arr01(1:end-2,1:end-2,1:end-2) = u_mag;
arr02(2:end-1,1:end-2,1:end-2) = u_mag;
arr03(3:end  ,1:end-2,1:end-2) = u_mag;
arr04(1:end-2,2:end-1,1:end-2) = u_mag;
arr05(2:end-1,2:end-1,1:end-2) = u_mag;
arr06(3:end  ,2:end-1,1:end-2) = u_mag;
arr07(1:end-2,3:end  ,1:end-2) = u_mag;
arr08(2:end-1,3:end  ,1:end-2) = u_mag;
arr09(3:end  ,3:end  ,1:end-2) = u_mag;

arr10(1:end-2,1:end-2,2:end-1) = u_mag;
arr11(2:end-1,1:end-2,2:end-1) = u_mag;
arr12(3:end  ,1:end-2,2:end-1) = u_mag;
arr13(1:end-2,2:end-1,2:end-1) = u_mag;
% Leave out centre array
arr14(3:end  ,2:end-1,2:end-1) = u_mag;
arr15(1:end-2,3:end  ,2:end-1) = u_mag;
arr16(2:end-1,3:end  ,2:end-1) = u_mag;
arr17(3:end  ,3:end  ,2:end-1) = u_mag;

arr18(1:end-2,1:end-2,3:end  ) = u_mag;
arr19(2:end-1,1:end-2,3:end  ) = u_mag;
arr20(3:end  ,1:end-2,3:end  ) = u_mag;
arr21(1:end-2,2:end-1,3:end  ) = u_mag;
arr22(2:end-1,2:end-1,3:end  ) = u_mag;
arr23(3:end  ,2:end-1,3:end  ) = u_mag;
arr24(1:end-2,3:end  ,3:end  ) = u_mag;
arr25(2:end-1,3:end  ,3:end  ) = u_mag;
arr26(3:end  ,3:end  ,3:end  ) = u_mag;

% Reshape all the data to one large array, using single element indexing. Array
% will have dimensions [prod(blank_size) x 26]
u_i = [ arr01(:) arr02(:) arr03(:) arr04(:) arr05(:) arr06(:) arr07(:) ...
         arr08(:) arr09(:) arr10(:) arr11(:) arr12(:) arr13(:) arr14(:) ...
          arr15(:) arr16(:) arr17(:) arr18(:) arr19(:) arr20(:) arr21(:) ...
           arr22(:) arr23(:) arr24(:) arr25(:) arr26(:)];

% For each row of med_filter, obtain the median value.
%   Note that NaNs are present in the data for two reasons:
%       1. Poor fit during the correlation process, leading to gaps in the
%           vector field (often where there aren't enough particles)
%       2. The NaN borders applied by expanding the arrays above.
%   In either case, we do not wish to include the NaN values in our computation
%   of the median. Fortunately MATLAB allows us to deal with this using a
%   statistics toolbox function...
u_med = nanmedian(u_i,2);

% So we have the median velocity surrounding each point.
% Now we unwrap the central array velocity distribution into a vector of the
% right size and shape:
arr_ctr = NaN(blank_size);
arr_ctr(2:end-1,2:end-1,2:end-1) = u_mag;
arr_ctr = arr_ctr(:);

% Calculate the residuals r_i for each of the surrounding points
r_i = abs(u_i - repmat(u_med,[1 26]));

% Calculate the median residual
r_med = nanmedian(r_i,2);

% Add an eps value to r_med (corresponding to mean noise level of the PIV data)
r_med = r_med + eps_PIV_noise;

% Get the difference between centre point velocities and median surrounding
% values
u_diff = abs(arr_ctr - u_med);

% Make the normalised median test, retaining any NaNs in the current data as
% being invalid.

% Debugging display...
%     disp('HERE__________________________')
%     isnanudiff = reshape(isnan(u_diff), blank_size)
%     isnanrmed = reshape(isnan(r_med), blank_size)
%     udiffOnrmed = reshape((u_diff./r_med), blank_size)
med_test = isnan(u_diff) | ((u_diff./r_med) > eps_threshold);

% Reshape the test array (required for indicating which vectors are invalid) and
% the r_med array (useful for determining stringency of detection). Crop both
% arrays to exclude the added boundary points
med_test = reshape(med_test, blank_size);
nmr   = reshape((u_diff./r_med), blank_size);
med_test = med_test(2:end-1,2:end-1,2:end-1);
nmr    = nmr(2:end-1,2:end-1,2:end-1);

end % end function med26

