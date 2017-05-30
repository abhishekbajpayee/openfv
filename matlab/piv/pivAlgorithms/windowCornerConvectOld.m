function [l0 l1 l2 l3 l4 l5 l6 l7 wConvection] = windowCornerConvect(l0, l1, l2, l3, l4, l5, l6, l7, guessField, dt, order, boundsX, boundsY, boundsZ)
%WINDOWCORNERCONVECT Convects window corner locations with an input flow field
%
% TODO: DOCUMENT PROPERLY
%
% loc 0 through loc 7 are arranged in X,Y,Z form, for a large number of windows:
% i.e. are [nWindows x 3] arrays.
%
% outputs are size [Ord*nW x 3] as vertically concatenated sets of [nW x 3] arrays i.e. [locsA;locsB] for different fields 
%
% guessField is in voxels/s units, in TomoPIV Toolbox format:
%     velocityField(pass).ux NB IN VOXEL/S UNITS FOR DISTANCE
%     velocityField(pass).uy
%     velocityField(pass).uz
%     velocityField(pass).nanMask
%     velocityField(pass).winCtrsVoxX
%     velocityField(pass).winCtrsVoxY
%     velocityField(pass).winCtrsVoxZ

%   Copyright (c) 2007-2015  Thomas H. Clark
% We use the winCtrsVox# fields instead of the winCtrsMM# fields to interpolate
% the velocity field onto the volume.


% Get displacement of fluid on the guessField grid in a time dt
if isstruct(guessField)
    uX = guessField.ux*dt;
    uY = guessField.uy*dt;
    uZ = guessField.uz*dt;
end

% For each of the corners...
for iCorner = 0:7
    
    
    % This is bloody ugly code. In a hurry though...
    switch iCorner
        case 0
            loc = l0;
        case 1
            loc = l1;
        case 2
            loc = l2;
        case 3
            loc = l3;
        case 4
            loc = l4;
        case 5
            loc = l5;
        case 6
            loc = l6;
        case 7
            loc = l7;
    end
    
    % Account for scalar guesses
    if ~isstruct(guessField)
        if numel(guessField) == 1
            % Guessfield must be a 1x3
            guessField = repmat(guessField,[1 3]);
        end
        dispX = guessField(1)*ones(size(loc,1),1);
        dispY = guessField(2)*ones(size(loc,1),1);
        dispZ = guessField(3)*ones(size(loc,1),1);
        
    else
        % Get the unit displacement in X,Y,Z at the required positions in
        % the grid. NB include ability to extrapolate the data, otherwise
        % we end up with NaNs in the guessed field
        [dispX] = nearestextrapdata3(guessField.winCtrsX(:), guessField.winCtrsY(:), guessField.winCtrsZ(:), uX(:), loc(:,1), loc(:,2), loc(:,3), 'nearest');
        [dispY] = nearestextrapdata3(guessField.winCtrsX(:), guessField.winCtrsY(:), guessField.winCtrsZ(:), uY(:), loc(:,1), loc(:,2), loc(:,3), 'nearest');
        [dispZ] = nearestextrapdata3(guessField.winCtrsX(:), guessField.winCtrsY(:), guessField.winCtrsZ(:), uZ(:), loc(:,1), loc(:,2), loc(:,3), 'nearest');
        
    end
    
    % Replicate the initial location and displacement matrices to reflect
    % output format
    if strcmpi(order,'first')
        reploc = [loc; loc];
        dispX = [-0.5*dispX; 0.5*dispX];
        dispY = [-0.5*dispY; 0.5*dispY];
        dispZ = [-0.5*dispZ; 0.5*dispZ];
    else
        reploc = [loc; loc; loc; loc];
        dispX = [-1.5*dispX; -0.5*dispX; 0.5*dispX; 1.5*dispX];
        dispY = [-1.5*dispY; -0.5*dispY; 0.5*dispY; 1.5*dispY];
        dispZ = [-1.5*dispZ; -0.5*dispZ; 0.5*dispZ; 1.5*dispZ];
    end
    
    % Use subfunction to reduce the displacement to an acceptable amount
    [newLoc] = checkOOB(reploc, dispX, dispY, dispZ, boundsX, boundsY, boundsZ, order);
    
    % Save to relevant variable
    switch iCorner
        case 0
            l0 = newLoc;
        case 1
            l1 = newLoc;
        case 2
            l2 = newLoc;
        case 3
            l3 = newLoc;
        case 4
            l4 = newLoc;
        case 5
            l5 = newLoc;
        case 6
            l6 = newLoc;
        case 7
            l7 = newLoc;
    end
            
            
end


% Get the new centre locations of the windows
newMeanLocation = (l0+l1+l2+l3+l4+l5+l6+l7)./8;

% Get the window displacement. NB positive displacement in one direction,
% negative in the other. Subtract to get the displacement from A to B
if strcmpi(order,'first')
    ind = size(newMeanLocation,1)/2;
    wConvection = newMeanLocation((ind+1):end,:) - newMeanLocation(1:ind,:);
else
    ind = size(newMeanLocation,1)/4;
    wConvection = newMeanLocation((2*ind+1):3*ind,:) - newMeanLocation(ind+1:2*ind,:);
end


end % end main function

function [newLoc] = checkOOB(reploc, dispX, dispY, dispZ, boundsX, boundsY, boundsZ, order)
    

    % Add the displacements, reducing if required by out of bounds errors
    newLoc = reploc+[dispX dispY dispZ];
    
    % Displacement factor
    for dispFac = 0.9:-0.1:0
        
        % Check for out of bounds errors
        xMsk = (newLoc(:,1) > max(boundsX)) | (newLoc(:,1) < min(boundsX));
        yMsk = (newLoc(:,2) > max(boundsY)) | (newLoc(:,2) < min(boundsY));
        zMsk = (newLoc(:,3) > max(boundsZ)) | (newLoc(:,3) < min(boundsZ));

        % It is not valid to reduce displacement on only one of the window pair
        % since that moves the effective position at which the velocity is
        % calculated. Thus we use an OR statement on the mask to move both (or
        % all four, in the second order case) windows closer to the original
        % position
        if strcmpi(order,'first')
            xMsk = xMsk(1:end/2) | xMsk(end/2+1:end);
            yMsk = yMsk(1:end/2) | yMsk(end/2+1:end);
            zMsk = zMsk(1:end/2) | zMsk(end/2+1:end);
            xMsk = [xMsk;xMsk]; %#ok<*AGROW>
            yMsk = [yMsk;yMsk];
            zMsk = [zMsk;zMsk];
        else
            ind = size(xMsk,1)/4;
            xMsk = xMsk(1:ind) | xMsk(ind+1:2*ind) | xMsk(2*ind+1:3*ind) | xMsk(3*ind+1:end);
            yMsk = yMsk(1:ind) | yMsk(ind+1:2*ind) | yMsk(2*ind+1:3*ind) | yMsk(3*ind+1:end);
            zMsk = zMsk(1:ind) | zMsk(ind+1:2*ind) | zMsk(2*ind+1:3*ind) | zMsk(3*ind+1:end);
            xMsk = [xMsk;xMsk;xMsk;xMsk];
            yMsk = [yMsk;yMsk;yMsk;yMsk];
            zMsk = [zMsk;zMsk;zMsk;zMsk];
        end
        
        % Reduce by dispfactor
        newLoc(xMsk,1) = reploc(xMsk,1) + dispFac*dispX(xMsk);
        newLoc(yMsk,2) = reploc(yMsk,2) + dispFac*dispY(yMsk);
        newLoc(zMsk,3) = reploc(zMsk,3) + dispFac*dispZ(zMsk);
    
    end % endfor
        
end



