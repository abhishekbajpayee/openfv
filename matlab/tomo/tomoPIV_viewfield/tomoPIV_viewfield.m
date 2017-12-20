function varargout = tomoPIV_viewfield(varargin)
% TOMOPIV_VIEWFIELD M-file for tomoPIV_viewfield.fig
%      TOMOPIV_VIEWFIELD, by itself, creates a new TOMOPIV_VIEWFIELD or raises the existing
%      singleton*.
%
%      H = TOMOPIV_VIEWFIELD returns the handle to a new TOMOPIV_VIEWFIELD or the handle to
%      the existing singleton*.
%
%      TOMOPIV_VIEWFIELD('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in TOMOPIV_VIEWFIELD.M with the given input arguments.
%
%      TOMOPIV_VIEWFIELD('Property','Value',...) creates a new TOMOPIV_VIEWFIELD or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before tomoPIV_viewfield_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to tomoPIV_viewfield_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

%   Copyright (c) 2007-2015  Thomas H. Clark
% Edit the above text to modify the response to help tomoPIV_viewfield

% Last Modified by GUIDE v2.5 28-Sep-2009 16:20:31

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @tomoPIV_viewfield_OpeningFcn, ...
                   'gui_OutputFcn',  @tomoPIV_viewfield_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT




% Opening And Output Functions
function tomoPIV_viewfield_OpeningFcn(hObject, eventdata, handles, varargin) %#ok<VANUS,INUSL>
handles.output = hObject;


% Determine the setup file we wish to use for the current plot, and load
% the setup structure
try
    handles.setup = tomoPIV_loadsetup();
    guidata(hObject, handles);
catch %#ok<CTCH>
    % If there is an error, it was probably handled in tomoPIV_loadsetup. We'll
    % just exit quietly from here
    close(hObject)
end



function varargout = tomoPIV_viewfield_OutputFcn(hObject, eventdata, handles)   %#ok<STOUT,INUSD>













% CALLBACKS



function pushbutton_color_Callback(hObject, eventdata, handles) %#ok<INUSL,DEFNU>
c = uisetcolor;
set(handles.pushbutton_color,'BackgroundColor',c)



function slider_alpha_Callback(hObject, eventdata, handles) %#ok<INUSL,DEFNU>
% Determine the alpha value
value = get(handles.slider_alpha,'Value');
% Update the text to the correct alpha value
set(handles.text_alphavalue,'String',num2str(value,'%01.2f'))



function pushbutton_addfield_Callback(hObject, eventdata, handles) %#ok<INUSL,DEFNU>

% Get Isovalue, Colour, Transparency and Plot Range
isovalue = str2double(get(handles.edit_isovalue,'String'));
colorspec = get(handles.pushbutton_color,'BackgroundColor');
alphavalue = get(handles.slider_alpha,'Value');
popup_selected = get(handles.popupmenu1,'Value');
switch popup_selected
    case 1
        plot_option = 'entirefield';
    case 2
        plot_option = 'boundingbox';
    case 3
        plot_option = 'central64';
    case 4
        plot_option = 'central32';
    case 5
        plot_option = 'central16';
end

    
% If plotting a bounding box, we don't need a volume. Plot the bounding box and
% return. Otherwise load a volume. 
if strcmpi('boundingbox',plot_option)
    
    % Make the axes current for plotting
    axes(handles.axes1); %#ok<MAXES>

    % Get the volume box details out of the setup structure and plot
    NVOX_X = numel(handles.setup.vox_X);
    NVOX_Y = numel(handles.setup.vox_Y);
    NVOX_Z = numel(handles.setup.vox_Z);
    volmat_x = [1 NVOX_X NVOX_X 1 1 1 NVOX_X NVOX_X NVOX_X NVOX_X NVOX_X NVOX_X 1 1 1 1];
    volmat_y = [1 1 NVOX_Y NVOX_Y 1 1 1 1 1 NVOX_Y NVOX_Y NVOX_Y NVOX_Y NVOX_Y NVOX_Y 1];
    volmat_z = [1 1 1 1 1 NVOX_Z NVOX_Z 1 NVOX_Z NVOX_Z 1 NVOX_Z NVOX_Z 1 NVOX_Z NVOX_Z];
    plot3(volmat_x,volmat_y,volmat_z,'k-')
    return
    
else
    % Load the field
    try
        voxels = tomoPIV_loadvoxels();
    catch  %#ok<CTCH>
        return % Quietly handle it when user cancels the operation
    end
    nvox_X = numel(handles.setup.vox_X);
    nvox_Y = numel(handles.setup.vox_Y);
    nvox_Z = numel(handles.setup.vox_Z);
    if numel(voxels)~=(nvox_X*nvox_Y*nvox_Z)
        warning('MATLAB:TomoPIVToolbox:InvalidInput','tomoPIV_viewfield: .mat file selected may be invalid, or contain batch results. Attempting to plot the first field of a batch...')
        voxels = voxels{1};
    end        
    voxels = reshape(voxels, nvox_Y, nvox_X, nvox_Z);
end

% If plotting the entire field:
if strcmp(plot_option,'entirefield')
    % THIS MAY NEED UPDATING TO LIMIT MEMORY REQUIREMENTS
    fv = isosurface(voxels,isovalue);
    axes(handles.axes1)
    ph = patch(fv);
    % Give the patch the appropiate colour and transparency
    set(ph,'EdgeAlpha',alphavalue,'FaceAlpha',alphavalue)
    set(ph,'CData',colorspec)
    set(ph,'EdgeColor',colorspec)
    set(ph,'FaceColor',colorspec)
    return
end

% Reduce field sizes:
switch plot_option
    case 'central64'
        disp('plotting central')
        X_inds = (1:64) + ceil(nvox_X/2);
        size(X_inds)
        Y_inds = (1:64) + ceil(nvox_Y/2);
        Z_inds = (1:64) + ceil(nvox_Z/2);
    case 'central32'
        X_inds = (1:32) + ceil(nvox_X/2);
        Y_inds = (1:32) + ceil(nvox_Y/2);
        Z_inds = (1:32) + ceil(nvox_Z/2);
    case 'central16'
        X_inds = (1:16) + ceil(nvox_X/2);
        Y_inds = (1:16) + ceil(nvox_Y/2);
        Z_inds = (1:16) + ceil(nvox_Z/2);
    otherwise
end

% Ensure that array bounds are not exceeded:
X_inds = X_inds((X_inds>=1) & (X_inds<=nvox_X));
Y_inds = Y_inds((Y_inds>=1) & (Y_inds<=nvox_Y));
Z_inds = Z_inds((Z_inds>=1) & (Z_inds<=nvox_Z));

% Crop the voxels volume:
voxels = voxels(Y_inds,X_inds,Z_inds);

% Create a meshgrid output and plot isosurface
[globX,globY,globZ] = meshgrid(single(X_inds),single(Y_inds),single(Z_inds));
fv = isosurface(globX,globY,globZ,voxels,isovalue);
axes(handles.axes1);
ph = patch(fv);
xlim([min(X_inds) max(X_inds)]);
ylim([min(Y_inds) max(Y_inds)]);
zlim([min(Z_inds) max(Z_inds)]);

% Give the patch the appropiate colour and transparency
set(ph,'EdgeAlpha',alphavalue,'FaceAlpha',alphavalue)
set(ph,'CData',colorspec)
set(ph,'EdgeColor',colorspec)
set(ph,'FaceColor',colorspec)








% UNUSED CREATION FUNCTIONS AND CALLBACKS


function slider_alpha_CreateFcn(hObject, eventdata, handles) %#ok<DEFNU,INUSD>
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
function edit_isovalue_Callback(hObject, eventdata, handles) %#ok<DEFNU,INUSD>
function edit_isovalue_CreateFcn(hObject, eventdata, handles) %#ok<DEFNU,INUSD>
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
function popupmenu1_Callback(hObject, eventdata, handles) %#ok<DEFNU,INUSD>
function popupmenu1_CreateFcn(hObject, eventdata, handles) %#ok<DEFNU,INUSD>
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
