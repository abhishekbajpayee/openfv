% -------------------------------------------------------------------
% Function to
% 
% - Read raw camera data for Tomo reconstruction
% - Accordingly read the appropriate calibration file (currently does
%   not check for error)
% - Then reconstruct using certain settings
% - Save the reconstructed set as an image stack along with Q and
%   time taken
% -------------------------------------------------------------------

function [vols] = processTomo(config_file)

%% Parse config_file

yaml_data = yaml.ReadYaml(config_file);

img_path = yaml_data.img_path;
save_path = yaml_data.save_path;
calib_file = yaml_data.calib_file;

if img_path(end) ~= '/'
    img_path = [img_path '/'];
end
if save_path(end) ~= '/'
    save_path = [save_path '/'];
end
if img_path(1) ~= '/'
    warning(['Path to images (img_path) does not seem to be absolute! ' ...
             'Code might fail to find datasets.']);
end

%% Read Calibration File

setupInfo = read_sa_calib(calib_file);

f = 1/setupInfo.scale;
voxels = [yaml_data.img_size{1,1} yaml_data.img_size{1,2}];
voxels(3) = size(yaml_data.zmin:f:yaml_data.zmax, 2)-1;
bounds = [-voxels(1)*0.5*f voxels(1)*0.5*f; ...
          -voxels(2)*0.5*f voxels(2)*0.5*f];
bounds(3,:) = [yaml_data.zmin yaml_data.zmax];

setup = sa_calib_to_tomo(setupInfo, voxels, bounds);

if isfield(yaml_data, 'draw_setup')
    if yaml_data.draw_setup
        drawSetup(setup, setupInfo);
    end
end

%% Reading Datasets

dirs = dir([img_path setupInfo.cam_names{1}]);

inds = [];
for i=1:size(dirs, 1)
    if strcmp(dirs(i).name, '.') | strcmp(dirs(i).name, '..')
        inds = [inds i];
    end
end

dirs(inds) = [];
nFrames = size(dirs, 1);

start_frame = 1;
end_frame = nFrames;

if (isfield(yaml_data, 'start_frame'))
    start_frame = yaml_data.start_frame;
    end_frame = yaml_data.end_frame;
end

for i=start_frame:end_frame
    
    display(['Reconstructing frame ' num2str(i) '...']);
    
    processedImgs = cell(setupInfo.num_cams, 1);
    
    for c=1:setupInfo.num_cams
        % disp([img_path setupInfo.cam_names{c} '/' dirs(i).name])
        processedImgs{c,1} = imread([img_path setupInfo.cam_names{c} '/' ...
                                     dirs(i).name]);
        % converting to double and scaling to [0,1]
        processedImgs{c,1} = double(processedImgs{c,1})/255.0;
    end
    
    %% Calling Tomo
        
    tomoOpts = defineTomoOptions('algorithm', yaml_data.alg, ...
                                 'nMartIters', 5, ...
                                 'nMfgIters', 5, ...
                                 'muMart', yaml_data.mart_mu);

    [vol, ctime] = tomo(setup, 1:setupInfo.num_cams, ...
                        processedImgs(:,1), tomoOpts);

    display(['Done reconstructing time step ' num2str(i)]);
    display(['Time taken: ' num2str(ctime)]);
    
    % rescaling reconstruction to [0,255]
    mx = max(max(max(vol)));
    vol = vol/mx;
    vol = uint8(vol*255);
    
    %% Saving Results
    
    frame_num = strsplit(dirs(i).name, '.');
    frame_num = frame_num{1};
    foldername = [save_path frame_num];

    save_tomo_result(vol, foldername, ctime);
    
end

%% Sending complete email

% % Modify these two lines to reflect
% % your account and password.
% myaddress = 'ehlfishycam@gmail.com';
% mypassword = 'ehlwave2013';

% setpref('Internet','E_mail',myaddress);
% setpref('Internet','SMTP_Server','smtp.gmail.com');
% setpref('Internet','SMTP_Username',myaddress);
% setpref('Internet','SMTP_Password',mypassword);

% props = java.lang.System.getProperties;
% props.setProperty('mail.smtp.auth','true');
% props.setProperty('mail.smtp.socketFactory.class', ...
%                   'javax.net.ssl.SSLSocketFactory');
% props.setProperty('mail.smtp.socketFactory.port','465');

% sendmail('ab9@mit.edu', 'Tomo Job Complete', 'Tomo Job Complete.');