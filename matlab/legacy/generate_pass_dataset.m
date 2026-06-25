%% GENERATE_PASS_DATASET Build a large labeled dataset for a DNN that predicts PASS configurations.
% Labels come from fmincon_pass (optimal PA x-positions + waveguide powers).
% Inputs are user coordinates and QoS. Run this script instead of many manual main_pass runs.
%
% MODE: This script is ONLY for PASS ML data (param.rand_sc=1 inside). For conventional
%       or thesis runs, use main_pass.m — see PARAMETER_AND_RUN_GUIDE.txt.
%
% Tips:
%   - Start with target_samples = 50 and fmincon_fast = true to estimate runtime.
%   - For production, use 1000–5000+ samples; run overnight.
%   - Outputs one pair of CSV files (input + output) and periodic checkpoint .mat files.

clear; clc; % Clear the workspace before starting dataset generation.

%% Dataset generation settings % Start user-configurable dataset parameters.
target_samples = 500; % Set how many feasible PASS samples to collect in this run.
param.R = 0.5; % Set the QoS target in bps/Hz for every sample in this dataset.
fmincon_fast = true; % Use true for faster fmincon (recommended for large datasets).
checkpoint_every = 50; % Save a checkpoint .mat file after this many new feasible samples.
max_attempts = 20 * target_samples; % Limit total tries to avoid an infinite loop.

%% System parameters (same defaults as main_pass.m) % Start physical and PASS parameters.
param.Mu = 3; % Set the number of users to three.
param.D = 10; % Set the square service-area side length in meters.
param.H = 5; % Set the transmitter height above the user plane in meters.
param.Nwg = 3; % Set the number of waveguides.
param.Nt = 3; % Set the number of pinching antennas per waveguide.
param.pbs = db2pow(20 - 30); % Convert the BS power budget from 20 dBm to watts.
param.pcircuit = db2pow(10 - 30); % Convert the circuit power from 10 dBm to watts.

param.rand_sc = 1; % Force PASS mode for dataset generation.
param.fmincon_fast = fmincon_fast; % Pass the fast-solver flag to fmincon_pass.m.

%% Waveguide geometry % Start PASS geometry (same as main_pass.m).
param.betay = zeros(param.Nwg, 1); % Allocate waveguide y-coordinates.
for ng = 1:param.Nwg % Loop over waveguides.
    param.betay(ng, 1) = -param.D/2 + (ng - 1)*param.D/param.Nwg + param.D/(2*param.Nwg); % Center each waveguide in its y-strip.
end % End the waveguide loop.
param.loc0 = [-param.D/2 * ones(param.Nwg, 1), param.betay(:), param.H * ones(param.Nwg, 1)]; % Store waveguide feed points.

param.f = 0.3; % Set carrier frequency in THz.
param.B = 1; % Use normalized bandwidth (bps/Hz).
param.c = 3e8; % Set speed of light in m/s.
param.lambda = param.c / (param.f * 10^12); % Compute wavelength in meters.
param.eta = (param.c / (4*pi*param.f*10^12))^2; % Compute path-gain constant.
param.spacing = param.lambda / 2; % Set minimum PA spacing to half a wavelength.
param.sigmanoise = db2pow(-174 - 30) * 10^9; % Set thermal noise power in watts.

%% Output folders and filenames % Start output path configuration.
script_folder = fileparts(mfilename('fullpath')); % Get the folder containing this script.
if isempty(script_folder) % Handle the case where the script path is unavailable.
    script_folder = pwd; % Fall back to the current working directory.
end % End the script-folder fallback.

csv_primary = fullfile(script_folder, 'csv_data'); % Prefer saving CSV files in csv_data.
csv_fallback = fullfile(tempdir, 'PASS_CSV'); % Use temp folder if csv_data is not writable.
csv_folder = ensureWritableFolder(csv_primary, csv_fallback); % Select a writable CSV folder.

ckpt_primary = fullfile(script_folder, 'dataset_checkpoints'); % Prefer checkpoint folder in the project.
ckpt_fallback = fullfile(tempdir, 'PASS_dataset_checkpoints'); % Use temp folder if needed.
ckpt_folder = ensureWritableFolder(ckpt_primary, ckpt_fallback); % Select a writable checkpoint folder.

timestamp = datestr(now, 'yyyy_mm_dd_HH_MM_SS'); % Build a unique timestamp for output files.
input_csv = fullfile(csv_folder, ['dataset_input_pass_ml_' timestamp '.csv']); % Input features CSV path.
output_csv = fullfile(csv_folder, ['dataset_output_pass_ml_' timestamp '.csv']); % Output labels CSV path.

nIn = param.Mu * 2 + 1; % Count input features: user x,y for each user plus QoS.
nOut = param.Nt * param.Nwg + param.Nwg; % Count outputs: all PA x positions plus waveguide powers.

input_cols = cell(1, nIn); % Allocate input column names.
for u = 1:param.Mu % Loop over users.
    input_cols{2*u - 1} = sprintf('user%d_x', u); % Name user x columns.
    input_cols{2*u} = sprintf('user%d_y', u); % Name user y columns.
end % End the user column loop.
input_cols{end} = 'QoS_R'; % Name the QoS column.

output_cols = cell(1, nOut); % Allocate output column names.
for wg = 1:param.Nwg % Loop over waveguides for PA position names.
    for n = 1:param.Nt % Loop over PAs on each waveguide.
        output_cols{(wg - 1)*param.Nt + n} = sprintf('PA%dx%d', n, wg); % Name each PA x column.
    end % End the PA loop.
end % End the waveguide loop for PA names.
for wg = 1:param.Nwg % Loop over waveguides for power column names.
    output_cols{param.Nt*param.Nwg + wg} = sprintf('power_wg%d', wg); % Name each waveguide power column.
end % End the power column loop.

%% Collection loop % Start feasible-sample collection.
nCollected = 0; % Count feasible samples stored so far.
nAttempts = 0; % Count total optimization attempts.
tStart = tic; % Start timer for progress reporting.

fprintf('Generating PASS ML dataset: target=%d feasible samples, QoS=%.2f, fast=%d\n', ...
    target_samples, param.R, fmincon_fast); % Print run configuration.
fprintf('Input CSV:  %s\n', input_csv); % Print input CSV path.
fprintf('Output CSV: %s\n\n', output_csv); % Print output CSV path.

while nCollected < target_samples && nAttempts < max_attempts % Continue until enough samples or attempt limit.
    nAttempts = nAttempts + 1; % Increment attempt counter.

    param.loc_u = zeros(param.Mu, 2); % Allocate user positions for this attempt.
    param.loc_u(:, 1) = param.D * rand(param.Mu, 1) - param.D/2; % Random user x inside the square area.
    param.loc_u(:, 2) = param.D * rand(param.Mu, 1) - param.D/2; % Random user y inside the square area.

    [Valu, betax_tx, p, r, ~] = fmincon_pass(param); % Run PASS optimization for one user layout.

    if Valu == 0 % Skip infeasible realizations that violate QoS or fail to converge.
        continue; % Try another random user layout.
    end % End the infeasibility check.

    nCollected = nCollected + 1; % Count one more feasible labeled sample.

    users_flat = reshape(param.loc_u.', 1, []); % Flatten user coordinates for the input row.
    in_row = [users_flat, param.R]; % Build one input feature row.
    pos_flat = reshape(betax_tx, 1, []); % Flatten optimized PA x-positions.
    out_row = [pos_flat, p(:).']; % Build one output label row.

    T_in = array2table(in_row, 'VariableNames', input_cols); % Convert input row to a table.
    T_out = array2table(out_row, 'VariableNames', output_cols); % Convert output row to a table.

    if nCollected == 1 % Write the first row with column headers.
        writetable(T_in, input_csv); % Create the input CSV with headers.
        writetable(T_out, output_csv); % Create the output CSV with headers.
    else % Append later rows without repeating headers (R2018b: no writetable WriteMode).
        dlmwrite(input_csv, in_row, '-append', 'delimiter', ',', 'precision', 16); % Append one input row.
        dlmwrite(output_csv, out_row, '-append', 'delimiter', ',', 'precision', 16); % Append one output row.
    end % End the first-row vs append logic.

  if mod(nCollected, 10) == 0 || nCollected == target_samples % Print progress every 10 samples and at the end.
        elapsed = toc(tStart); % Read elapsed time in seconds.
        rate = nCollected / max(elapsed, 1e-6); % Estimate feasible samples per second.
        fprintf('  %d / %d feasible (attempts=%d, %.2f samples/min)\n', ...
            nCollected, target_samples, nAttempts, rate * 60); % Report progress.
    end % End the progress print.

    if mod(nCollected, checkpoint_every) == 0 % Save a checkpoint on a fixed interval.
        ckpt_file = fullfile(ckpt_folder, sprintf('checkpoint_pass_%s_n%d.mat', timestamp, nCollected)); % Build checkpoint path.
        save(ckpt_file, 'nCollected', 'nAttempts', 'param', 'input_csv', 'output_csv'); % Save lightweight checkpoint metadata.
        fprintf('  Checkpoint saved: %s\n', ckpt_file); % Report checkpoint path.
    end % End the checkpoint block.
end % End the collection while loop.

%% Summary % Start final reporting.
fprintf('\nDone. Feasible samples collected: %d / %d\n', nCollected, target_samples); % Print final sample count.
fprintf('Total attempts: %d\n', nAttempts); % Print total attempts.
fprintf('Input CSV:  %s\n', input_csv); % Repeat input CSV path.
fprintf('Output CSV: %s\n', output_csv); % Repeat output CSV path.

if nCollected < target_samples % Warn when the attempt limit stopped collection early.
    warning('Target not reached. Increase max_attempts or relax QoS param.R.'); % Suggest fixes for low yield.
end % End the incomplete-dataset warning.

fprintf('\nNext steps for DNN training:\n'); % Print guidance for machine-learning workflow.
fprintf('  1. Load both CSV files in Python/MATLAB.\n'); % Step 1.
fprintf('  2. Split into train/validation/test (e.g. 70/15/15).\n'); % Step 2.
fprintf('  3. Train network: input -> PASS positions + powers.\n'); % Step 3.
fprintf('  4. At inference, use DNN prediction instead of fmincon_pass for speed.\n'); % Step 4.
