function [disp_test] = PIV_3d_maxsnr(snr, max_snr)
%PIV_3D_MAXSNR Vector field validation using max signal to noise ratio
%   Simple approach: Where velocities (displacements) exceed the limiting
%   amount in x, y, or z component, they are labelled as spurious.
%   
%
% Syntax:  
%       [disp_test] = PIV_3d_maxsnr(snr, max_snr)
%
% Inputs:
%       snr     [ny, nx, nz]    3D array containing signal to noise ratios
%
%       max_snr [1 x 1]         Maximum snr allowable
%
% Outputs:
%
%       disp_test   [ny nx nz]  logical array 
%                               The same size as input snr array. True
%                               indicates the the vector is invalid,
%                               according to the above criterion.
%
%
% Examples:
%       see PIV_3d_validatefield.m for example usage in context.
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
% Created:          11 July 2011

%   Copyright (c) 2007-2015  Thomas H. Clark

% logical test:
disp_test = (snr > max_snr);















