function folderPath = ensureWritableFolder(primaryFolder, fallbackFolder)
%ENSUREWRITABLEFOLDER Return a folder that MATLAB can create files inside.

if nargin < 2 || isempty(fallbackFolder) % Check whether the caller omitted the fallback folder.
    fallbackFolder = fullfile(tempdir, 'PASS_output'); % Use a generic temporary PASS output folder as the fallback.
end % End the fallback-folder default check.

folderPath = tryPrepareFolder(primaryFolder); % Try to create and validate the preferred output folder first.

if ~isempty(folderPath) % Check whether the preferred folder is usable.
    return; % Return immediately because the preferred folder is writable.
end % End the preferred-folder success check.

warning('Primary output folder is not writable: %s. Using fallback folder: %s', primaryFolder, fallbackFolder); % Explain the fallback behavior.

folderPath = tryPrepareFolder(fallbackFolder); % Try to create and validate the fallback output folder.

if isempty(folderPath) % Check whether even the fallback folder is not usable.
    error('Cannot create a writable output folder. Tried primary folder "%s" and fallback folder "%s".', primaryFolder, fallbackFolder); % Stop with both attempted paths.
end % End the fallback-folder failure check.

end % End the ensureWritableFolder function.

function folderPath = tryPrepareFolder(candidateFolder)
%TRYPREPAREFOLDER Create a folder if needed and verify write permission.

folderPath = ''; % Start with an empty result, meaning the candidate folder has not yet been validated.

try % Protect folder creation and write-testing from permission errors.
    if ~exist(candidateFolder, 'dir') % Check whether the candidate folder does not exist yet.
        [created, msg] = mkdir(candidateFolder); % Try to create the candidate folder.
        if ~created % Check whether MATLAB failed to create the folder.
            error('%s', msg); % Throw MATLAB's mkdir message so the caller can fall back.
        end % End the mkdir failure check.
    end % End the folder-existence check.

    probeFile = [tempname(candidateFolder) '.tmp']; % Build a temporary file path inside the candidate folder.
    fid = fopen(probeFile, 'w'); % Try to open the temporary file for writing.

    if fid == -1 % Check whether MATLAB could not open the file.
        error('MATLAB cannot write a test file in this folder.'); % Throw a permission-style error.
    end % End the fopen failure check.

    fclose(fid); % Close the temporary test file.
    delete(probeFile); % Remove the temporary test file after validating write permission.

    folderPath = candidateFolder; % Mark the candidate folder as usable.
catch % Catch permission, path, sync-lock, and other filesystem errors.
    folderPath = ''; % Return an empty result so the caller can try the fallback folder.
end % End the protected folder-validation block.

end % End the tryPrepareFolder helper function.
