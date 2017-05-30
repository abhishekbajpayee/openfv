function [velocityStruct] = PIV_3d_matlab(fieldA, fieldB, dt, pivOpts)
%PIV_3D_MATLAB Particle Image Velocimetry using 3D scalar intensity fields
%   Performs windowed cross correlation of fieldA (particle field at time t)
%   with fieldB (particle field at time t+dt).
%
%   The cross correlation is performed using 3D FFTs and either gaussian or 
%   whittaker peak finding algorithm to determine the correlation displacement.
%   Displacement values are scaled by the time in seconds dt to give
%   output velocities in units voxels/s.
%
%   Window size can be varied. Multi-pass analysis is available, with varying
%   window size and false vector removal options. Successive passes use the
%   previous pass as a first guess, and a window-shifting algorithm is used.
%   Window deformation is not available.
%
% Syntax:  
%       velocityStructure = PIV_3d(fieldA, fieldB, dx, dt, pivOpts)
%
% Inputs:
%
%       fieldA, fieldB          [nVoxY x nVoxZ x nVoxZ] single
%                                               Contains the reconstructed
%                                               intensity field at a point in
%                                               time, where A,B,C,D represent a
%                                               consecutive time series, each
%                                               frame separated by time dt.
%
%       dt                      [1 x 1] double  The time in seconds between
%                                               successive reconstruction
%                                               instances.
%
%       pivOpts                 structure       Contains the options for
%                                               specifying which PIV algorithm
%                                               to use and the behaviour of the
%                                               algorithm, false vector
%                                               selection parameters, multipass
%                                               options etc. This structure is
%                                               created using the 
%                                                   definePIVOptions() function.
%                                               See help definePIVOptions
%                                               for further details.
%
% Outputs:
%
%       velocityStructure       structure(nPasses)
%                                               A structure containing the
%                                               intermediate and final velocity
%                                               fields produced by PIV_3d. This
%                                               structure can be used in further
%                                               plotting, analysis and
%                                               prostprocessing of the results.
%
% Note on Units:
%
%       Ouptut velocities are in voxels/second. The outputs (in the velocity
%       structure) containing the window centres are in voxel units relative to
%       the local reconstruction array. Use the PIV_3d routine to call this
%       function, as PIV_3d updates the velocity structure to give the outputs
%       in correct units.
%
% References:
%   [1] Raffel M. Willert C. Wereley S. and Kompenhans J. Particle Image
%       Velocimetry A Practical Guide (Second Edition). Springer 2007 
%       ISBN 978-3-540-72307-3
%
%   [2] Thomas M. Misra S. Kambhamettu C. and Kirby J.T. (2005)
%       A robust motion estimation algorithm for PIV
%
% Future Improvements:
%
%   [1] Implementation of a first guess (currently excluded from the code)
%
%   [2] For passes > 1 where window translation is used, the code to adjust
%       the amount of translation (in the event of array bounds exceeded
%       problems) should be re-expressed in terms of the displacement, to make
%       it clearer.
%
%   [3] Full implementation of the phase-correlation based false vector checking
%       
% Other m-files required:   none
% Subfunctions:             weightingMatrix Computes the 3D cross correlation
%                           weighting matrix (see ref. 1)
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
% Revison History:          16 April 2011       Created, reformulated from
%                                               the original version of PIV_3d.m
%                                               Documentation improved and minor
%                                               modifications made to
%                                               inputs/outputs for compatibility
%                                               with new version of PIV_3d.m
%                           17 April 2011       Updated documentation

%   Copyright (c) 2007-2015  Thomas H. Clark



%% PRELIMINARY CALCULATIONS AND SETUP


% Initialise the results structure
emptycell      = cell(pivOpts.npasses,1);
velocityStruct = struct('ux',           emptycell, ...
                        'uy',           emptycell, ...
                        'uz',           emptycell, ...
                        'indicator',    emptycell, ...
                        'peak_locs',    emptycell, ...
                        'peak_vals',    emptycell, ...
                        'peak_void',    emptycell, ...
                        'win_ctrs_x',   emptycell, ...
                        'win_ctrs_y',   emptycell, ...
                        'win_ctrs_z',   emptycell);
   
% Size of voxels arrays
nvox_X = size(fieldA,2);
nvox_Y = size(fieldA,1);
nvox_Z = size(fieldA,3);



%% LOOP FOR DIFFERENT PASSES

for pass = 1:pivOpts.npasses
    
    % DIsplay progress
    disp(['PIV_3d_matlab.m: Pass ' num2str(pass) ' of ' num2str(pivOpts.npasses)])
    
    
    
    % SORT WINDOW DISCRETISATION
    
    % Window sizes for current pass...
    wsize = pivOpts.wsize(pass,:);
    
    % Get window centre coords in X, Y, Z
    min_coord = wsize/2 + 0.5 + pivOpts.edge_cut;
    max_coord = [nvox_X nvox_Y nvox_Z] - wsize/2 +0.5 - pivOpts.edge_cut;
    win_spacing = wsize * (1 - (pivOpts.overlap(pass)/100));
    win_ctrs_x = min_coord(1):win_spacing(1):max_coord(1);
    win_ctrs_y = min_coord(2):win_spacing(2):max_coord(2);
    win_ctrs_z = min_coord(3):win_spacing(3):max_coord(3);
    %   So windows are spaced win_spacing number of voxels apart. The coordinate
    %   of the first window in each direction is min_coord, and the maximum
    %   coordinate possible is max_coord.
    
    % Number of windows in each direction
    n_windows = [numel(win_ctrs_x) numel(win_ctrs_y) numel(win_ctrs_z)];
    
    % Indices into voxels array of the starting corners (corner with lowest
    % X,Y,Z) of each interrogation window:
    start_inds_x = round(win_ctrs_x - wsize(1)/2 + 0.5);
    start_inds_y = round(win_ctrs_y - wsize(2)/2 + 0.5);
    start_inds_z = round(win_ctrs_z - wsize(3)/2 + 0.5);
    
    % Indices into voxels array of finishing corners (corner with highest X,Y,Z)
    % for each interrogation window:
    end_inds_x = start_inds_x + wsize(1) - 1;
    end_inds_y = start_inds_y + wsize(2) - 1;
    end_inds_z = start_inds_z + wsize(3) - 1;
    
    % The second window (to be cross-correlated with the first window whose
    % indices are determined above) is able to translate, based on a guessed
    % velocity field (from previous passes). Each window in the field is
    % translated based on the previous pass displacement from that point in the
    % field, so rather than setting up vector arrays of indices, we need to set
    % up multidimensional arrays to contain translated window indices.
    % Setting up for zero translation:
    [grid_start_x grid_start_y grid_start_z] = meshgrid(start_inds_x, start_inds_y, start_inds_z);
    [grid_end_x   grid_end_y   grid_end_z]   = meshgrid(end_inds_x,   end_inds_y,   end_inds_z);
    
    % NOTE POTENTIAL BUG HERE. We should check that end_inds_i do not exceed
    % nvox_i. If they do, then remove the final window. The remaining windows
    % should then be shifted to centre on the volume. Vectors affected will be
    % win_ctrs, n_windows, start_inds and end_inds. Hasn't occurred so far after
    % months of testing...
    
    
    % Form a meshgrid of x, y, z window centre points. This is the volume grid
    % on which the velocity data Ux, Uy and Uz will be calculated.
    % Display can be made using the MATLAB intrinsic function:
    %            >>  coneplot(mesh_x, mesh_y, mesh_z, Ux, Uy, Uz, 'nointerp')
    % NB the axes of the resultant plot will be in voxel units, not mm, unless
    % the mesh arrays are scaled appropriately first.
    [mesh_x mesh_y mesh_z] = meshgrid(win_ctrs_x,win_ctrs_y,win_ctrs_z);
        
    
    
    % DETERMINE THE TRANSLATION FROM PREVIOUS PASSES 
    
    if pass > 1
                
        % Smooth the velocity field from the previous pass
        ux = smooth3(ux,'gaussian',[3 3 3],3);
        uy = smooth3(uy,'gaussian',[3 3 3],3);
        uz = smooth3(uz,'gaussian',[3 3 3],3);
        
        % Interpolate velocity field from the previous pass onto current grid.
        % NB have to use spline due to occasional requirement for extrapolation.
        disp_x = interp3(oldmsh_x,oldmsh_y,oldmsh_z,ux,mesh_x,mesh_y,mesh_z,'spline');
        disp_y = interp3(oldmsh_x,oldmsh_y,oldmsh_z,uy,mesh_x,mesh_y,mesh_z,'spline');
        disp_z = interp3(oldmsh_x,oldmsh_y,oldmsh_z,uz,mesh_x,mesh_y,mesh_z,'spline');
        
        % These first guess values give us the translation directly, although we
        % need to use them as indices, so we should round (toward zero - which 
        % conservatively reduces the window displacement amount slightly). We
        % translate the frame A windows by -half the window displacement, and 
        % frame B windows by +half...
        half_dx = fix(disp_x/2);
        half_dy = fix(disp_y/2);
        half_dz = fix(disp_z/2);
        
        % Define the start and end indices of translated windows. Since the
        % velocity field varies across the volume, we cannot express these as
        % vectors of indices, but rather have to have a whole field of
        % indices (so each window is displaced by the appropriate amount in
        % each direction)
        trA_start_x = grid_start_x - half_dx;
        trA_start_y = grid_start_y - half_dy;
        trA_start_z = grid_start_z - half_dz;
        trB_start_x = grid_start_x + half_dx;
        trB_start_y = grid_start_y + half_dy;
        trB_start_z = grid_start_z + half_dz;
        
        trA_end_x = grid_end_x - half_dx;
        trA_end_y = grid_end_y - half_dy;
        trA_end_z = grid_end_z - half_dz;
        trB_end_x = grid_end_x + half_dx;
        trB_end_y = grid_end_y + half_dy;
        trB_end_z = grid_end_z + half_dz;
        
        % If translation of the window puts us beyond the edge of the field,
        % then we must reduce the translation in order to prevent out of bounds
        % access errors. The best we can get is translating the window right to
        % the edge of the field instead of outside it.
        
        % For index exceeding maximum bounds...
        xmax = nvox_X - pivOpts.edge_cut;
        ymax = nvox_Y - pivOpts.edge_cut;
        zmax = nvox_Z - pivOpts.edge_cut;
        
        x_mask = trA_end_x > xmax;
        y_mask = trA_end_y > ymax;
        z_mask = trA_end_z > zmax;
        trA_start_x(x_mask) = trA_start_x(x_mask) - trA_end_x(x_mask) + xmax;
        trA_start_y(y_mask) = trA_start_y(y_mask) - trA_end_y(y_mask) + ymax;
        trA_start_z(z_mask) = trA_start_z(z_mask) - trA_end_z(z_mask) + zmax;
        trA_end_x(x_mask) = xmax;
        trA_end_y(y_mask) = ymax;
        trA_end_z(z_mask) = zmax;
        
        x_mask = trB_end_x > xmax;
        y_mask = trB_end_y > ymax;
        z_mask = trB_end_z > zmax;
        trB_start_x(x_mask) = trB_start_x(x_mask) - trB_end_x(x_mask) + xmax;
        trB_start_y(y_mask) = trB_start_y(y_mask) - trB_end_y(y_mask) + ymax;
        trB_start_z(z_mask) = trB_start_z(z_mask) - trB_end_z(z_mask) + zmax;
        trB_end_x(x_mask) = xmax;
        trB_end_y(y_mask) = ymax;
        trB_end_z(z_mask) = zmax;
        
        % For indices less than lower bounds...
        x_mask = trA_start_x <= pivOpts.edge_cut;
        y_mask = trA_start_y <= pivOpts.edge_cut;
        z_mask = trA_start_z <= pivOpts.edge_cut;
        trA_end_x(x_mask) = trA_end_x(x_mask) - trA_start_x(x_mask) + pivOpts.edge_cut + 1;
        trA_end_y(y_mask) = trA_end_y(y_mask) - trA_start_y(y_mask) + pivOpts.edge_cut + 1;
        trA_end_z(z_mask) = trA_end_z(z_mask) - trA_start_z(z_mask) + pivOpts.edge_cut + 1;
        trA_start_x(x_mask) = pivOpts.edge_cut + 1;
        trA_start_y(y_mask) = pivOpts.edge_cut + 1;
        trA_start_z(z_mask) = pivOpts.edge_cut + 1;
        
        x_mask = trB_start_x <= pivOpts.edge_cut;
        y_mask = trB_start_y <= pivOpts.edge_cut;
        z_mask = trB_start_z <= pivOpts.edge_cut;
        trB_end_x(x_mask) = trB_end_x(x_mask) - trB_start_x(x_mask) + pivOpts.edge_cut + 1;
        trB_end_y(y_mask) = trB_end_y(y_mask) - trB_start_y(y_mask) + pivOpts.edge_cut + 1;
        trB_end_z(z_mask) = trB_end_z(z_mask) - trB_start_z(z_mask) + pivOpts.edge_cut + 1;
        trB_start_x(x_mask) = pivOpts.edge_cut + 1;
        trB_start_y(y_mask) = pivOpts.edge_cut + 1;
        trB_start_z(z_mask) = pivOpts.edge_cut + 1;
        
        % Update displacement arrays...
        disp_x = trB_end_x - trA_end_x;
        disp_y = trB_end_y - trA_end_y;
        disp_z = trB_end_z - trA_end_z;
        
    else
        % No translation on first pass. Set tr_start_i and tr_end_i to the grid
        % arrays:
        trA_start_x = grid_start_x;
        trA_start_y = grid_start_y;
        trA_start_z = grid_start_z;
        trA_end_x = grid_end_x;
        trA_end_y = grid_end_y;
        trA_end_z = grid_end_z;
        
        trB_start_x = grid_start_x;
        trB_start_y = grid_start_y;
        trB_start_z = grid_start_z;
        trB_end_x = grid_end_x;
        trB_end_y = grid_end_y;
        trB_end_z = grid_end_z;
        
    end
    
    % Save the current pass mesh to be used for the interpolation next pass
    oldmsh_x = mesh_x;
    oldmsh_y = mesh_y;
    oldmsh_z = mesh_z;
    
   
    
    % OBTAIN CROSS-CORRELATION ELEMENTS
    
    % Weighting matrix
    [weight] = weightingmatrix(wsize);
       
    % Initialise arrays for results
    pk_locs_x = zeros([3 n_windows(2) n_windows(1) n_windows(3)]);
    pk_locs_y = pk_locs_x;
    pk_locs_z = pk_locs_x;
    
    
    % LOOP OVER ALL WINDOWS IN Y, X, THEN Z DIRECTIONS
    
    %   Due to the convention of the tomoPIV toolbox for storing intensity
    %   fields, the global Y direction corresponds with the first dimension
    %   of the intensity arrays (see 'structure definitions.m' for sign convs).
    %   Thus, we cycle through the windows in this order to improve efficiency.
    for yctr = 1:n_windows(2)
        
        for xctr = 1:n_windows(1)
            
            for zctr = 1:n_windows(3)
                % disp(['Coords: ' num2str([yctr xctr zctr])])
                
                % x, y, z index vectors of the current window, displaced
                % backward by half the first guess.
                x_inds_1 = trA_start_x(yctr, xctr, zctr):1:trA_end_x(yctr, xctr, zctr);
                y_inds_1 = trA_start_y(yctr, xctr, zctr):1:trA_end_y(yctr, xctr, zctr);
                z_inds_1 = trA_start_z(yctr, xctr, zctr):1:trA_end_z(yctr, xctr, zctr);
                
                % x, y, z index vectors of the current window, displaced forward
                % by half the first guess:
                x_inds_2 = trB_start_x(yctr, xctr, zctr):1:trB_end_x(yctr, xctr, zctr);
                y_inds_2 = trB_start_y(yctr, xctr, zctr):1:trB_end_y(yctr, xctr, zctr);
                z_inds_2 = trB_start_z(yctr, xctr, zctr):1:trB_end_z(yctr, xctr, zctr);
                
                % Get the current interrogation window out of the field
                window1 = fieldA(y_inds_1, x_inds_1, z_inds_1);
                window2 = fieldB(y_inds_2, x_inds_2, z_inds_2);
                
                
                   
                % Take 3D FFTs of the windows.
                %   The execution time for fft depends on the length of the 
                %   transform. It is fastest for powers of two. It is almost as 
                %   fast for lengths that have only small prime factors.
                %   It is typically several times slower for lengths that are 
                %   prime or which have large prime factors.
                %
                %   NB is it worth padding the FFT? probably not! Just make IW
                %   sizes sensible!
                %
                %   NB single-precision data. The FFT should work OK with that.
                w_1 = fftn(window1);
                w_2 = fftn(window2);
                
                % Cross-correlate the data
                corr = w_2.*conj(w_1);
                
                % Smooth the correlation volume using:
                %       - gaussian filter kernel
                %       - of size 3x3x3
                %       - of standard deviation 0.65
                % This is EXTREMELY slow but produces very nice results.
%                 corr = smooth3(corr,'gaussian',[3 3 3],3);
                
                % Apply zero-whitening FIR filters (presented in reference [2])
                % to calculate the phase correlation volume
%                 phase_corr = corr./(abs(w_1).*abs(w_2));
            
                % Inverse FFT to obtain real domain correlation volume.
                real_corr = real(ifftn(corr));
%                 real_phase_corr = real(ifftn(phase_corr));
%                 real_corr = real_corr.*real_phase_corr;
                
                % Shift the zero (DC) frequency to the centre of the volume
                real_corr = fftshift(real_corr);
                
                           
                % Divide by the weighting matrix
                %   - debiases the correlation peak (see p.137 Ref [1]).
                real_corr = real_corr./weight;
                
                % Find the peak locations.
                % The facility to detect multiple peaks in a correlation window
                % is allowed for.
                %   pk_loc is a 3xnPeaks array. Each column contains ux, uy, uz 
                %   displacments (in voxel units). Column 1 corresponds to the
                %   primary peak, column 2 to the secondary peak, etc...
                tlo = ceil(size(real_corr)/2)-ceil(size(real_corr)/4);
                thi = ceil(size(real_corr)/2)+floor(size(real_corr)/4);
                switch lower(pivOpts.peakFinder)
                    case 'gaussian'
                        [pk_loc] = PIV_3d_peaklocate(real_corr(tlo(1):thi(1),tlo(2):thi(2),tlo(3):thi(3)),3);
                    case 'whittaker'
                        [pk_loc] = PIV_3d_sincpeaklocate(real_corr(tlo(1):thi(1),tlo(2):thi(2),tlo(3):thi(3)),3);
                    case 'cubic'
                        [pk_loc] = PIV_3d_cubicpeaklocate(real_corr(tlo(1):thi(1),tlo(2):thi(2),tlo(3):thi(3)));
                    otherwise
                        error('MATLAB:TomoPIVToolbox:UnavailableOption','Incorrect specifier for peak location method. Try ''cubic'' or ''gaussian''')
                end
                        
                % We rearrange them to store in 3 4-D arrays (dimensions are
                % nPeaks, n_windows(2), n_windows(1), n_windows(3) )
                pk_locs_x(:, yctr, xctr, zctr) = pk_loc(1,:)+(tlo(2)-1);
                pk_locs_y(:, yctr, xctr, zctr) = pk_loc(2,:)+(tlo(1)-1);
                pk_locs_z(:, yctr, xctr, zctr) = pk_loc(3,:)+(tlo(3)-1);
                
                % Number of peaks retrieved by the peak location algorithm
                nPeaks = size(pk_locs_x,1);
                
                % PLOT OUTPUT
                plotting = pivOpts.plot;
                if plotting  && (yctr == 4) && (xctr == 4) %&& (zctr == 1)
                    
                    % Set up figure with 3 subplots:
                    fh = figure(206);
                    clf
                    figname = ['PIV_3d_matlab Debug Plot: yctr = ' num2str(yctr) ' xctr = ' num2str(xctr) ' zctr = ' num2str(zctr)];
                    set(fh,'NumberTitle', 'off');
                    set(fh,'Name',figname);
                    subplot(1,3,1);
                    
                    % Plot the particle distributions:
                    fld_h_1 =  patch(isosurface(window1,0.0001));
                    fld_h_2 =  patch(isosurface(window2,0.0001));
                    set(fld_h_1,'FaceColor','blue','EdgeColor','none');
                    set(fld_h_2,'FaceColor','red','EdgeColor','none');
                    title('Particle Distributions')
    
                    ah2 = subplot(1,3,2);
                    bug = reshape(real_corr,[numel(real_corr) 1]);
                    bug(bug <= 2*eps('single')) = [];
                    hist(ah2, bug,100);
                    title('histogram of real, nonzero values in the correlation volume')
                                        
                    % Plot actual points of correlation peaks
                    subplot(1,3,3);
                    %   First peak blue, second peak red, third green...
                    plot3(pk_loc(1,1),pk_loc(2,1),pk_loc(3,1),'b.','MarkerSize',5)
                    hold on
                    if nPeaks > 1
                        plot3(pk_loc(1,2),pk_loc(2,2),pk_loc(3,2),'r.','MarkerSize',5)
                    end
                    if nPeaks > 2
                        plot3(pk_loc(1,3),pk_loc(2,3),pk_loc(3,3),'g.','MarkerSize',5)
                    end
                    xlim([0 wsize(1)])
                    ylim([0 wsize(2)])
                    zlim([0 wsize(3)])
                    
                    % Isosurfaces of the correlation volume
                    loc_inds = round(pk_loc);
                    plot_real_corr = double(real_corr) + rand(double(size(real_corr)))*0.001;
                    plot_real_corr = plot_real_corr./max(plot_real_corr(:));
                    if sum(isnan(loc_inds(:,1))) == 0
                        isomat = double(plot_real_corr(tlo(1):thi(1),tlo(2):thi(2),tlo(3):thi(3)));
                        isoval = 0.2.*max(isomat(:));
                        pk1_iso_h = patch(isosurface(isomat,isoval));
                        set(pk1_iso_h,'FaceColor','blue','EdgeColor','none');
                    end
%                     if sum(isnan(loc_inds(:,2))) == 0
%                         pk_2_value = real_corr(loc_inds(1,2),loc_inds(2,2),loc_inds(3,2));
%                         pk2_iso_h = patch(isosurface(double(real_corr),double(0.99*pk_2_value)));
%                         set(pk2_iso_h,'FaceColor','red','EdgeColor','none');
%                     end
%                     if sum(isnan(loc_inds(:,3))) == 0
%                         pk_3_value = real_corr(loc_inds(1,3),loc_inds(2,3),loc_inds(3,3));
%                         pk3_iso_h = patch(isosurface(double(real_corr),double(0.99*pk_3_value)));
%                         set(pk3_iso_h,'FaceColor','green','EdgeColor','none');
%                     end
                    disp('Paused, press any key')
                    pause
                    
                end % end if plotting
                
            end % end for zctr
        end % end for xctr
    end % end for yctr
    
    
    % Correct peak locations for:
    %   - window centre coordinates (u relative to window centre)
    pk_locs_x = pk_locs_x - (wsize(1)/2) - 1;
    pk_locs_y = pk_locs_y - (wsize(2)/2) - 1;
    pk_locs_z = pk_locs_z - (wsize(3)/2) - 1;
    
    % ranslation of windows. Not necessary on the first pass.
%     warning('Authors note - possible error where velocity guess field is applied at this point in the code')
    if pass > 1
        pk_locs_x = pk_locs_x + repmat(reshape(disp_x, [1 size(disp_x)]),[nPeaks 1 1 1]);
        pk_locs_y = pk_locs_y + repmat(reshape(disp_y, [1 size(disp_y)]),[nPeaks 1 1 1]);
        pk_locs_z = pk_locs_z + repmat(reshape(disp_z, [1 size(disp_z)]),[nPeaks 1 1 1]);
    end
    
    % Cross-correlation is complete for the current pass. Perform vector
    % validation on the current field:
    [ux uy uz] = PIV_3d_vectorcheck(n_windows, pk_locs_x, pk_locs_y, pk_locs_z, pass, pivOpts);
    
    % Gap fill where NaNs remain in the velocity arrays.
    [ux uy uz nan_mask] = PIV_3d_gapfill(mesh_x, mesh_y, mesh_z, ux, uy, uz, pivOpts.gapFillMethod);
    if sum(sum(sum(isnan(ux) | isnan(uy) | isnan(uz)))) > 0
        error('MATLAB:TomoPIVToolbox:InvalidInput','Gap filling not working, as all velocities NaN. Check inputs and try again.')
    end
        
    % Store the results of this pass to a structure.
    velocityStruct(pass).ux = ux/dt;
    velocityStruct(pass).uy = uy/dt;
    velocityStruct(pass).uz = uz/dt;
    velocityStruct(pass).nan_mask = nan_mask;
    velocityStruct(pass).pk_locs_x = pk_locs_x;
    velocityStruct(pass).pk_locs_y = pk_locs_y;
    velocityStruct(pass).pk_locs_z = pk_locs_z;
    velocityStruct(pass).win_ctrs_x = win_ctrs_x;
    velocityStruct(pass).win_ctrs_y = win_ctrs_y;
    velocityStruct(pass).win_ctrs_z = win_ctrs_z;
    

end % End for pass = 1:npasses






end % END FUNCTION PIV_3d_matlab


%% SUBFUNCTIONS


function [weight] = weightingmatrix(window_size)

% Calculates the cross-correlation weighting matrix
%
% TODO: Crap FOR loop should be replaced with vectorised code!
m = window_size(2);
n = window_size(1); % NB swapped around for y and x consistency with main routine
k = window_size(3);
wdisttocen =  (  ((m+1)/2)^2  +  ((n+1)/2)^2  +  ((k+1)/2)^2  )^0.5 ;

weight = zeros(m,n,k);
for ii = 1:m
    for jj = 1:n
        for kk = 1:k
            wdistfromcen = ( ((ii-((m+1)/2))^2) + ((jj-((n+1)/2))^2) +  ((kk-((k+1)/2))^2) )^0.5;
            weight(ii,jj,kk) = 1 - (wdistfromcen/wdisttocen);
        end
    end
end

weight(weight <= 0) = 0.01;

end % end function
