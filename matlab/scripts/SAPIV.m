% Function to run PIV on multiple reconstructed frames in a directory
% based on config file output by OpenFV SA reconstruction

function velocityData = SAPIV(config_file)

% extract data from yaml config file
yaml_data = yaml.ReadYaml(config_file);

data_path = yaml_data.data_path;
save_path = yaml_data.piv_save_path;
f = yaml_data.pix_per_mm;
nPasses = yaml_data.passes;

wSize = [yaml_data.windows{1,1}, yaml_data.windows{1,2}, ...
         yaml_data.windows{1,3};
         yaml_data.windows{2,1}, yaml_data.windows{2,2}, ...
         yaml_data.windows{2,3};
         yaml_data.windows{3,1}, yaml_data.windows{3,2}, ...
         yaml_data.windows{3,3}];

overlap = [yaml_data.overlap{1,1}, yaml_data.overlap{1,2}, ...
           yaml_data.overlap{1,3};
           yaml_data.overlap{2,1}, yaml_data.overlap{2,2}, ...
           yaml_data.overlap{2,3};
           yaml_data.overlap{3,1}, yaml_data.overlap{3,2}, ...
           yaml_data.overlap{3,3}];


pivOpts = definePIVOptions('nPasses', nPasses, ...
                           'wSize', wSize, ...
                           'overlap', overlap, ...
                           'fetchType', [3 3 3], ...
                           'algorithm', 'fmexpar');

% get directories in folder
contents = dir(data_path);

% filter dir contents to remove config file, . and ..
datasets = [];
for i = 1:size(contents, 1)
    if ~(contents(i).isdir == 0 | strcmp(contents(i).name, '.') | ...
         strcmp(contents(i).name, '..'))
        datasets = [datasets; contents(i)];
    end
end

nFrames = size(datasets, 1);
for i = 1:nFrames-1

    frames = {datasets(i).name, datasets(i+1).name};
    disp(['Processing frames ' frames{1} ' and ' ...
                 frames{2}]);
    
    results_path = [save_path frames{1}];
    if exist(results_path, 'dir')
        if exist([results_path '/3DPIV_results.mat'], 'file')
            warning(['Warning: PIV results already exist for frame ' frames{1} '. Will ' ...
                     'be overwritten!']);
        end
    end
    
    try
        
        velocityField = runPIV(data_path, frames, pivOpts, f, 1.0);

        mkdir(results_path);
        save([results_path '/3DPIV_results.mat'], ...
                 'velocityField');
    
    catch ERROR
        
        ERROR
        disp(['Could not process frames ' frames{1} ' and ' ...
                 frames{2}]);
        
    end
        
end

