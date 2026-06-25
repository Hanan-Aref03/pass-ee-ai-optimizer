% DATA Export a CSV dataset from a saved MAT result file. % Explain the script purpose.
% Conventional output contains only optimized powers. % Explain the conventional dataset convention.
% PASS output contains optimized PA x-positions followed by waveguide powers. % Explain the PASS dataset convention.

[file, path] = uigetfile('*.mat', 'Select a saved results file'); % Ask the user to select a MAT result file.
if isequal(file, 0) % Check whether the file-selection dialog was cancelled.
    error('No MAT file selected.'); % Stop because there is no input file to process.
end % End the file-selection check.

load(fullfile(path, file)); % Load the selected MAT file into the workspace.

isConventional = (param.rand_sc == 0); % Detect whether the loaded file is conventional mode.
isPASS = (param.rand_sc == 1); % Detect whether the loaded file is PASS mode.

if ~(isConventional || isPASS) % Reject loaded files with unsupported rand_sc values.
    error('Unsupported mode in selected MAT file. Expected param.rand_sc=0 or 1.'); % Stop because the saved result mode is not supported.
end % End the mode-validation condition.

if exist('valid_counts', 'var') && ~isempty(valid_counts) % Check whether the MAT file saved valid sample counts.
    num_samples = valid_counts(1); % Use the saved feasible-sample count for the first QoS value.
else % Handle older result files that do not contain valid_counts.
    num_samples = size(loc_u_all, 3); % Fall back to the third dimension of loc_u_all.
end % End the sample-count detection.

if num_samples == 0 % Check whether the selected file contains no feasible samples.
    error('The selected MAT file contains no feasible samples.'); % Stop because there is nothing to export.
end % End the empty-result check.

Mu = param.Mu; % Store the number of users in a shorter local variable.
inputs = zeros(num_samples, Mu*2 + 1); % Allocate input rows: user coordinates plus QoS.

if isConventional % Configure conventional output size.
    outputs = zeros(num_samples, param.Na); % Store only optimized powers, not fixed constant antenna coordinates.
else % Configure PASS output size.
    outputs = zeros(num_samples, param.Nt*param.Nwg + param.Nwg); % Store optimized PA x-positions plus waveguide powers.
end % End the output-size configuration.

for s = 1:num_samples % Loop over feasible samples.
    users_flat = reshape(loc_u_all(:, :, s, 1).', 1, []); % Flatten users as [user1_x,user1_y,user2_x,user2_y,...].
    inputs(s, :) = [users_flat, param.R]; % Store user coordinates and QoS value in the input matrix.

    powers_flat = pbs_all(:, s, 1)'; % Read the saved powers as a row vector.
    if isConventional % Build conventional output row.
        outputs(s, :) = powers_flat; % Store only the optimized fixed-antenna powers.
    else % Build PASS output row.
        pos_flat = reshape(betax_all(:, :, s, 1), 1, []); % Flatten all optimized PA x-positions for the selected sample.
        outputs(s, :) = [pos_flat, powers_flat]; % Store PA x-positions followed by waveguide powers.
    end % End the output-row construction.
end % End the sample loop.

input_cols = cell(1, Mu*2 + 1); % Allocate input-column names.
for i = 1:Mu % Loop over all users.
    input_cols{2*i - 1} = sprintf('user%d_x', i); % Name each user x-coordinate column.
    input_cols{2*i} = sprintf('user%d_y', i); % Name each user y-coordinate column.
end % End the input-column loop.
input_cols{end} = 'QoS_R'; % Name the QoS input column.

if isConventional % Build conventional output-column names.
    output_cols = cell(1, param.Na); % Allocate one column name per optimized power.
    for a = 1:param.Na % Loop over fixed BS antennas.
        output_cols{a} = sprintf('power_ant%d', a); % Name the optimized power column.
    end % End the conventional output-column loop.
    mode_name = 'conventional_fixed_bs'; % Set the conventional dataset filename suffix.
else % Build PASS output-column names.
    output_cols = cell(1, param.Nt*param.Nwg + param.Nwg); % Allocate column names for PA positions and powers.
    for wg = 1:param.Nwg % Loop over waveguides.
        for n = 1:param.Nt % Loop over PAs on each waveguide.
            output_cols{(wg - 1)*param.Nt + n} = sprintf('PA%dx%d', n, wg); % Name each PA x-position column.
        end % End the PA loop.
    end % End the waveguide loop for PA positions.
    for wg = 1:param.Nwg % Loop over waveguides for power names.
        output_cols{param.Nt*param.Nwg + wg} = sprintf('power_wg%d', wg); % Name each waveguide power column.
    end % End the waveguide-power loop.
    mode_name = 'pass'; % Set the PASS dataset filename suffix.
end % End the output-column-name construction.

T_in = array2table(inputs, 'VariableNames', input_cols); % Convert the input matrix to a table.
T_out = array2table(outputs, 'VariableNames', output_cols); % Convert the output matrix to a table.
T_full = [T_in, T_out]; % Concatenate input and output tables into one full dataset.

filename = fullfile(path, ['dataset_full_' mode_name '.csv']); % Build the output CSV filename in the same folder as the MAT file.
writetable(T_full, filename); % Write the full dataset to CSV.
fprintf('Dataset saved to: %s\n', filename); % Print the saved CSV path.
