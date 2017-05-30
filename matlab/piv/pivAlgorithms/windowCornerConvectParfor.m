function [l0 l1 l2 l3 l4 l5 l6 l7 wConvection] = windowCornerConvect2(l0, l1, l2, l3, l4, l5, l6, l7, guessField, dt, order, boundsX, boundsY, boundsZ)
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



% Input locations to an array
locs(1:size(l0,1),1:size(l0,2),1) = l0;
locs(1:size(l0,1),1:size(l0,2),2) = l1;
locs(1:size(l0,1),1:size(l0,2),3) = l2;
locs(1:size(l0,1),1:size(l0,2),4) = l3;
locs(1:size(l0,1),1:size(l0,2),5) = l4;
locs(1:size(l0,1),1:size(l0,2),6) = l5;
locs(1:size(l0,1),1:size(l0,2),7) = l6;
locs(1:size(l0,1),1:size(l0,2),8) = l7;
if strcmpi(order,'second');
    locsOut = zeros(size(locs,1)*4,size(locs,2),8);
else
    locsOut = zeros(size(locs,1)*2,size(locs,2),8);
end

% For each of the corners...
parfor iCorner = 1:8
    
    locsOut(:,:,iCorner) = subWindowConvect(locs(:,:,iCorner), dt, guessField, order, boundsX, boundsY, boundsZ);
    
end

l0 = locsOut(:,:,1);
l1 = locsOut(:,:,2);
l2 = locsOut(:,:,3);
l3 = locsOut(:,:,4);
l4 = locsOut(:,:,5);
l5 = locsOut(:,:,6);
l6 = locsOut(:,:,7);
l7 = locsOut(:,:,8);

% Get the new centre locations of the windows
newMeanLocation = (l0+l1+l2+l3+l4+l5+l6+l7)./8;

% Get the window displacement. NB positive displacement in one direction,
% negative in the other. Subtract to get the displacement from A to B
if strcmpi(order,'first')
    ind = size(newMeanLocation,1)/2;
    wConvection = newMeanLocation((ind+1):end,:) - newMeanLocation(1:ind,:);
else
    ind = size(newMeanLocation,1)/4;
%     wConvection = newMeanLocation((2*ind+1):3*ind,:) - newMeanLocation(ind+1:2*ind,:);
    wConvection = (newMeanLocation((2*ind+1):3*ind,:) - newMeanLocation(1:ind,:))/2;
end


end % end main function







function [newLoc] = subWindowConvect(loc, dt, guessField, order, boundsX, boundsY, boundsZ)


    % Get displacement of fluid on the guessField grid in a time dt
    if isstruct(guessField)
        uX = guessField.ux*dt;
        uY = guessField.uy*dt;
        uZ = guessField.uz*dt;
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
        [dispX] = nearestextrapdata3(guessField.winCtrsX(:), guessField.winCtrsY(:), guessField.winCtrsZ(:), uX(:), loc(:,1), loc(:,2), loc(:,3), 'linear');
        [dispY] = nearestextrapdata3(guessField.winCtrsX(:), guessField.winCtrsY(:), guessField.winCtrsZ(:), uY(:), loc(:,1), loc(:,2), loc(:,3), 'linear');
        [dispZ] = nearestextrapdata3(guessField.winCtrsX(:), guessField.winCtrsY(:), guessField.winCtrsZ(:), uZ(:), loc(:,1), loc(:,2), loc(:,3), 'linear');
        
    end
    
    % Replicate the initial location and displacement matrices to reflect
    % output format
%     if strcmpi(order,'first')
%         reploc = [loc; loc];
%         dispX = [-0.5*dispX; 0.5*dispX];
%         dispY = [-0.5*dispY; 0.5*dispY];
%         dispZ = [-0.5*dispZ; 0.5*dispZ];
%     else
%         reploc = [loc; loc; loc; loc];
%         dispX = [-1.5*dispX; -0.5*dispX; 0.5*dispX; 1.5*dispX];
%         dispY = [-1.5*dispY; -0.5*dispY; 0.5*dispY; 1.5*dispY];
%         dispZ = [-1.5*dispZ; -0.5*dispZ; 0.5*dispZ; 1.5*dispZ];
%     end
    
    %% FIRST MOVEMENT
    reploc = [loc; loc];
    dispX = [-0.5*dispX; 0.5*dispX];
    dispY = [-0.5*dispY; 0.5*dispY];
    dispZ = [-0.5*dispZ; 0.5*dispZ];
    
    %% CONVECT AGAIN
    if strcmpi(order,'second')
        % Account for scalar guesses
        if ~isstruct(guessField)
            if numel(guessField) == 1
                % Guessfield must be a 1x3
                guessField = repmat(guessField,[1 3]);
            end
            dispX2 = guessField(1)*ones(2*size(loc,1),1);
            dispY2 = guessField(2)*ones(2*size(loc,1),1);
            dispZ2 = guessField(3)*ones(2*size(loc,1),1);
        else
            convLoc = reploc + [dispX dispY dispZ];
            [dispX2] = nearestextrapdata3(guessField.winCtrsX(:), guessField.winCtrsY(:), guessField.winCtrsZ(:), uX(:), convLoc(:,1), convLoc(:,2), convLoc(:,3), 'nearest');
            [dispY2] = nearestextrapdata3(guessField.winCtrsX(:), guessField.winCtrsY(:), guessField.winCtrsZ(:), uY(:), convLoc(:,1), convLoc(:,2), convLoc(:,3), 'nearest');
            [dispZ2] = nearestextrapdata3(guessField.winCtrsX(:), guessField.winCtrsY(:), guessField.winCtrsZ(:), uZ(:), convLoc(:,1), convLoc(:,2), convLoc(:,3), 'nearest');
        end
        dispX2(1:end/2) = -1*dispX2(end/2+1:end);
        dispY2(1:end/2) = -1*dispY2(end/2+1:end);
        dispZ2(1:end/2) = -1*dispZ2(end/2+1:end);
        reploc = [loc;loc;loc;loc];
        dispX = [dispX2(1:end/2)+dispX(1:end/2); dispX; dispX2(end/2+1:end)+dispX(end/2+1:end)];
        dispY = [dispY2(1:end/2)+dispY(1:end/2); dispY; dispY2(end/2+1:end)+dispY(end/2+1:end)];
        dispZ = [dispZ2(1:end/2)+dispZ(1:end/2); dispZ; dispZ2(end/2+1:end)+dispZ(end/2+1:end)];
    end
    
    % Use subfunction to reduce the displacement to an acceptable amount
    [newLoc] = checkOOB(reploc, dispX, dispY, dispZ, boundsX, boundsY, boundsZ, order);
    
end % end subfunction subWindowConvect







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
    
    % Perform final check
    xMsk = (newLoc(:,1) > max(boundsX)) | (newLoc(:,1) < min(boundsX));
    yMsk = (newLoc(:,2) > max(boundsY)) | (newLoc(:,2) < min(boundsY));
    zMsk = (newLoc(:,3) > max(boundsZ)) | (newLoc(:,3) < min(boundsZ));
    if any(xMsk) || any(yMsk) || any(zMsk)
        error('argh! there are still windows which will violate bounds. this is a known bug. they must be masked out.... try a different window size and/or overlap as a workaround')
    end
end



