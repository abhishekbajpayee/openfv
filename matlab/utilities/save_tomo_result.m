% Function to save tomo reconstruction to file such that
% it can be read later for PIV

function [result] = save_tomo_result(vol, foldername, ctime)

if exist(foldername, 'dir')
            
    display (['Folder ' foldername ' already exists! Skipping to ' ...
                        'avoid overwriting...']);
    result = 0;
    
else
    
    display (['Saving reconstruction to ' foldername '...']);
    mkdir(foldername);
        
    % write computation time to file
    % f = fopen([foldername '/t.txt'], 'w');
    % fprintf(f, '%f', ctime);
        
    % dump stack
    for f=1:size(vol, 3)
        imname = sprintf('%03.f.tif', f);
        imwrite(vol(:,:,f), [foldername '/' imname]);
    end
    
    result = 1;
    
end

end