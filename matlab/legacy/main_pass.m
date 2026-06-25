clear all; clc; % Clear the MATLAB workspace and command window before starting the simulation.

ct = 50; % Set the target number of feasible channel/user realizations to collect.

%% System parameters % Start the section that defines all simulation parameters.
param.Mu = 3; % Set the number of users to three.
param.D = 10; % Set the square service-area side length in meters.
param.H = 5; % Set the transmitter height above the user plane in meters.

param.Nwg = 3; % Fixed system size: 3 waveguides in PASS; conventional uses Na=Nwg for the same antenna count.
param.Na = param.Nwg; % Conventional mode: 3 fixed BS antennas (no waveguides); Na equals Nwg for a fair element-count comparison.
param.Nt = 3; % Number of pinching antennas per waveguide (PASS mode only; not used in conventional mode).

param.pbs = db2pow(20 - 30); % Convert the BS power budget from 20 dBm to watts.
param.pcircuit = db2pow(10 - 30); % Convert the circuit power from 10 dBm to watts.

%% Mode selection % Start the section that chooses the simulation mode (two modes only).
% param.rand_sc = 0  ->  conventional fixed BS (power optimized by fmincon_fpa)
% param.rand_sc = 1  ->  PASS (joint PA positions + powers optimized by fmincon_pass)
% For ML dataset generation use generate_pass_dataset.m instead — see PARAMETER_AND_RUN_GUIDE.txt
param.rand_sc = 1; % Set to 0 for conventional or 1 for PASS before running this script.

isConventional = (param.rand_sc == 0); % Detect conventional fixed-BS-antenna mode.
isPASS = (param.rand_sc == 1); % Detect PASS mode with joint fmincon optimization.

if ~(isConventional || isPASS) % Reject any rand_sc value other than 0 or 1.
    error('Unsupported mode. Use param.rand_sc=0 (conventional) or param.rand_sc=1 (PASS).'); % Stop with a clear configuration message.
end % End the mode-validation condition.

if isConventional % Configure dimensions and labels for conventional mode.
    param.mode_name = 'conventional_fixed_bs'; % Store a text label for result and CSV filenames.
    numPowerStreams = param.Na; % One optimized power value per fixed BS antenna (3 antennas, power-only optimization).
    posRows = param.Na; % Store one row per fixed BS antenna.
    posCols = 2; % Store fixed BS antenna [x,y] coordinates (vertical lambda/2 array; no waveguides).
else % Configure dimensions and labels for PASS mode.
    param.mode_name = 'pass'; % Store a text label for result and CSV filenames.
    numPowerStreams = param.Nwg; % One optimized power value per waveguide.
    posRows = param.Nt; % Store one row per PA on each waveguide.
    posCols = param.Nwg; % Store one column per waveguide.
end % End the mode-specific dimension configuration.

%% Waveguide geometry (PASS mode only; ignored by conventional fixed-BS channel model) % Start PASS geometry section.
param.betay = zeros(param.Nwg, 1); % Allocate the y-coordinate vector of the waveguides for PASS.
for ng = 1:param.Nwg % Loop over each waveguide.
    param.betay(ng, 1) = -param.D/2 + (ng - 1)*param.D/param.Nwg + param.D/(2*param.Nwg); % Place each waveguide at the center of its y-strip.
end % End the waveguide-position loop.

param.loc0 = [-param.D/2 * ones(param.Nwg, 1), param.betay(:), param.H * ones(param.Nwg, 1)]; % Define each waveguide feed point as [x0,y0,z0].

%% Noise, RF, and propagation constants % Start the section that defines physical-layer constants.
param.sigmanoise = db2pow(-174 - 30) * 10^9; % Convert -174 dBm/Hz thermal noise to watts over 1 GHz.
param.f = 0.3; % Set the carrier frequency in THz.
param.B = 1; % Use normalized bandwidth so rates are expressed in bps/Hz.
param.c = 3e8; % Set the speed of light in meters per second.
param.lambda = param.c / (param.f * 10^12); % Compute the free-space wavelength in meters.
param.eta = (param.c / (4*pi*param.f*10^12))^2; % Compute the free-space gain coefficient used by the channel model.
param.spacing = param.lambda / 2; % Set the minimum spacing between adjacent PAs to half a wavelength.

R_vec = [0.1]; % Define the QoS rate requirement values to simulate in bps/Hz.

%% Preallocate result containers % Start the section that reserves memory for results.
Cee_all = zeros(ct, length(R_vec)); % Store the energy-efficiency value for each feasible realization and QoS value.
p_tot_all = zeros(ct, length(R_vec)); % Store total consumed power for each feasible realization and QoS value.
pbs_all = zeros(numPowerStreams, ct, length(R_vec)); % Store optimized power values for each transmit stream.
rate_all = zeros(ct, param.Mu, length(R_vec)); % Store the achieved rate of each user.
betax_all = zeros(posRows, posCols, ct, length(R_vec)); % Store antenna or PA position data for each realization.
loc_u_all = zeros(param.Mu, 2, ct, length(R_vec)); % Store user coordinates for every feasible realization.
valid_counts = zeros(length(R_vec), 1); % Store how many feasible samples were collected per QoS value.
attempts = zeros(length(R_vec), 1); % Store how many infeasible attempts occurred per QoS value.

max_attempts_per_qos = 10 * ct; % Limit retries to avoid an infinite loop when the QoS target is infeasible.

%% Main loop over QoS values % Start the simulation loop over all QoS targets.
for j = 1:length(R_vec) % Iterate over each QoS value.

    param.R = R_vec(j); % Store the current QoS value inside the parameter structure.

    Cee = zeros(ct, 1); % Allocate temporary energy-efficiency values for the current QoS value.
    p_tot = zeros(ct, 1); % Allocate temporary total-power values for the current QoS value.
    pbs = zeros(numPowerStreams, ct); % Allocate temporary power values for the current QoS value.
    rate = zeros(ct, param.Mu); % Allocate temporary user-rate values for the current QoS value.
    betax_tmp = zeros(posRows, posCols, ct); % Allocate temporary antenna-position values for the current QoS value.
    loc_u_tmp = zeros(param.Mu, 2, ct); % Allocate temporary user-location values for the current QoS value.

    i1 = 1; % Initialize the feasible-sample counter.
    total_attempts = 0; % Initialize the total-attempt counter.

    while i1 <= ct && total_attempts < max_attempts_per_qos % Continue until enough feasible samples are found or the attempt limit is reached.
        total_attempts = total_attempts + 1; % Count the current realization attempt.

        param.loc_u = zeros(param.Mu, 2); % Allocate a user-location matrix for the current attempt.
        param.loc_u(:, 1) = param.D * rand(param.Mu, 1) - param.D/2; % Randomly generate user x-coordinates inside the square area.
        param.loc_u(:, 2) = param.D * rand(param.Mu, 1) - param.D/2; % Randomly generate user y-coordinates inside the square area.

        [Valu, betax_tx, p, r, p_total] = fmincon_main(param); % Run conventional or PASS optimization for one realization.

        if Valu == 0 % Treat zero energy efficiency as an infeasible realization.
            attempts(j) = attempts(j) + 1; % Count the infeasible attempt for the current QoS value.
            continue; % Skip storage and try another random user realization.
        end % End the infeasibility check.

        Cee(i1) = Valu; % Store the energy efficiency of the feasible realization.
        rate(i1, :) = r(:)'; % Store the per-user achievable rates as a row vector.
        p_tot(i1) = p_total; % Store the total consumed power.
        pbs(:, i1) = p(:); % Store the power vector.

        betax_tmp(:, :, i1) = betax_tx; % Store fixed BS [x,y] coordinates (conventional) or PA x-positions (PASS).

        loc_u_tmp(:, :, i1) = param.loc_u; % Store the user coordinates of the feasible realization.
        i1 = i1 + 1; % Move to the next feasible-sample index.
    end % End the while loop for the current QoS value.

    valid_counts(j) = i1 - 1; % Store the number of feasible samples actually collected.

    if valid_counts(j) < ct % Warn when the attempt limit prevented collection of all requested samples.
        warning('Only %d feasible realizations found out of %d for R = %.2f.', valid_counts(j), ct, param.R); % Report the incomplete sample count.
    end % End the incomplete-sampling warning condition.

    Cee_all(:, j) = Cee; % Copy temporary energy-efficiency values to the final container.
    p_tot_all(:, j) = p_tot; % Copy temporary total-power values to the final container.
    pbs_all(:, :, j) = pbs; % Copy temporary power values to the final container.
    rate_all(:, :, j) = rate; % Copy temporary rate values to the final container.
    betax_all(:, :, :, j) = betax_tmp; % Copy temporary antenna coordinates to the final container.
    loc_u_all(:, :, :, j) = loc_u_tmp; % Copy temporary user coordinates to the final container.

    script_folder = fileparts(mfilename('fullpath')); % Get the folder that contains this main script.
    if isempty(script_folder) % Check whether MATLAB could not resolve the script folder.
        script_folder = pwd; % Fall back to the current MATLAB working folder.
    end % End the script-folder fallback check.

    primary_results_folder = fullfile(script_folder, 'results'); % Prefer saving results inside the project results folder.
    fallback_results_folder = fullfile(tempdir, 'PASS_results'); % Use the system temporary folder when the project folder is not writable.
    save_path = ensureWritableFolder(primary_results_folder, fallback_results_folder); % Create or select a writable result-output folder.

    filename = fullfile(save_path, ['results_' param.mode_name '_R_' num2str(param.R) '_' datestr(now, 'yyyy_mm_dd_HH_MM_SS') '.mat']); % Build the timestamped MAT filename.

    save(filename, 'Cee_all', 'p_tot_all', 'pbs_all', 'rate_all', 'betax_all', 'loc_u_all', 'valid_counts', 'attempts', 'param'); % Save all simulation outputs and parameters.

    fprintf('Results saved to: %s\n', filename); % Print the location of the saved MAT file.
end % End the loop over QoS values.

fprintf('Mean EE over feasible realizations:\n'); % Print a header for the mean energy-efficiency summary.
for j = 1:length(R_vec) % Loop over QoS values for summary printing.
    nValid = valid_counts(j); % Read the number of feasible samples for the current QoS value.
    if nValid > 0 % Check whether at least one feasible sample exists.
        fprintf('R = %.2f: %.6g\n', R_vec(j), mean(Cee_all(1:nValid, j))); % Print the mean EE over feasible samples only.
    else % Handle the case where no feasible samples were found.
        fprintf('R = %.2f: no feasible realization found\n', R_vec(j)); % Print that no feasible realization exists.
    end % End the feasible-sample summary condition.
end % End the summary loop.

%% Convert first QoS result to CSV for machine-learning export % Start CSV generation from the first QoS result.
num_samples = valid_counts(1); % Read the number of feasible samples for the first QoS value.

if num_samples > 0 % Generate CSV files only when at least one feasible sample exists.
    input_features = zeros(num_samples, param.Mu*2 + 1); % Allocate inputs: user coordinates plus QoS requirement.

    if isConventional % Configure the output size for conventional mode.
        output_features = zeros(num_samples, param.Na); % Store only optimized powers because fixed antenna coordinates are constants.
    else % Configure the output size for PASS mode.
        output_features = zeros(num_samples, param.Nt*param.Nwg + param.Nwg); % Store optimized PA positions plus waveguide powers.
    end % End the mode-specific output-size configuration.

    for s = 1:num_samples % Loop over feasible samples.
        users_flat = reshape(loc_u_all(:, :, s, 1).', 1, []); % Flatten users as [user1_x,user1_y,user2_x,user2_y,...].
        input_features(s, :) = [users_flat, R_vec(1)]; % Store the input row with user coordinates and QoS.

        powers_flat = pbs_all(:, s, 1)'; % Read the power values as a row vector.
        if isConventional % Build the conventional output row.
            output_features(s, :) = powers_flat; % Store only the optimized powers returned by fmincon_fpa.
        else % Build the PASS output row.
            pos_flat = reshape(betax_all(:, :, s, 1), 1, []); % Flatten all optimized PA x-positions from all waveguides.
            output_features(s, :) = [pos_flat, powers_flat]; % Store PA positions followed by waveguide powers.
        end % End the mode-specific output-row construction.
    end % End the CSV sample loop.

    input_cols = cell(1, param.Mu*2 + 1); % Allocate the input-column-name cell array.
    for u = 1:param.Mu % Loop over users to name their coordinate columns.
        input_cols{2*u - 1} = sprintf('user%d_x', u); % Name the x-coordinate column of the current user.
        input_cols{2*u} = sprintf('user%d_y', u); % Name the y-coordinate column of the current user.
    end % End the input-column-name loop.
    input_cols{end} = 'QoS_R'; % Name the final input column as the QoS target.

    if isConventional % Build conventional output-column names.
        output_cols = cell(1, param.Na); % Allocate one output column per optimized antenna power.
        for a = 1:param.Na % Loop over fixed BS antennas.
            output_cols{a} = sprintf('power_ant%d', a); % Name the optimized power column of each antenna.
        end % End the conventional output-column-name loop.
    else % Build PASS output-column names.
        output_cols = cell(1, param.Nt*param.Nwg + param.Nwg); % Allocate output columns for PA positions and waveguide powers.
        for wg = 1:param.Nwg % Loop over waveguides.
            for n = 1:param.Nt % Loop over PAs on the current waveguide.
                output_cols{(wg - 1)*param.Nt + n} = sprintf('PA%dx%d', n, wg); % Name each PA x-position column.
            end % End the PA loop.
        end % End the waveguide loop for PA-position names.
        for wg = 1:param.Nwg % Loop over waveguides again for power columns.
            output_cols{param.Nt*param.Nwg + wg} = sprintf('power_wg%d', wg); % Name each waveguide power column.
        end % End the power-column-name loop.
    end % End the mode-specific output-column-name construction.

    T_in = array2table(input_features, 'VariableNames', input_cols); % Convert the input matrix to a table with named columns.
    T_out = array2table(output_features, 'VariableNames', output_cols); % Convert the output matrix to a table with named columns.

    csv_primary_folder = fullfile(script_folder, 'csv_data'); % Prefer saving CSV files inside the project csv_data folder.
    csv_fallback_folder = fullfile(tempdir, 'PASS_CSV'); % Use the system temporary folder when the project folder is not writable.
    csv_save_path = ensureWritableFolder(csv_primary_folder, csv_fallback_folder); % Create or select a writable CSV output folder.

    timestamp = datestr(now, 'yyyy_mm_dd_HH_MM_SS'); % Build a timestamp string for unique CSV filenames.
    csv_input_file = fullfile(csv_save_path, sprintf('dataset_input_%s_%s.csv', param.mode_name, timestamp)); % Build the input CSV filename.
    csv_output_file = fullfile(csv_save_path, sprintf('dataset_output_%s_%s.csv', param.mode_name, timestamp)); % Build the output CSV filename.

    try % Protect CSV writing from permission or file-lock errors.
        writetable(T_in, csv_input_file); % Write the input feature table to CSV.
        writetable(T_out, csv_output_file); % Write the output label table to CSV.
        fprintf('Input CSV saved to: %s\n', csv_input_file); % Report the input CSV path.
        fprintf('Output CSV saved to: %s\n', csv_output_file); % Report the output CSV path.
    catch ME % Handle CSV write failures in the primary folder.
        warning('Failed to write CSV files to %s: %s', csv_save_path, ME.message); % Warn about the primary-folder failure.
        home_csv = fullfile(getenv('USERPROFILE'), 'Documents', 'PASS_CSV'); % Build a fallback CSV folder in the user Documents directory.
        [success, msg] = mkdir(home_csv); % Try to create the fallback folder.
        if success % Check whether the fallback folder was created.
            csv_input_file = fullfile(home_csv, sprintf('dataset_input_%s_%s.csv', param.mode_name, timestamp)); % Rebuild the input CSV path in the fallback folder.
            csv_output_file = fullfile(home_csv, sprintf('dataset_output_%s_%s.csv', param.mode_name, timestamp)); % Rebuild the output CSV path in the fallback folder.
            try % Try writing the CSV files again in the fallback folder.
                writetable(T_in, csv_input_file); % Write the input CSV in the fallback folder.
                writetable(T_out, csv_output_file); % Write the output CSV in the fallback folder.
                fprintf('Input CSV saved to (fallback): %s\n', csv_input_file); % Report the fallback input CSV path.
                fprintf('Output CSV saved to (fallback): %s\n', csv_output_file); % Report the fallback output CSV path.
            catch ME2 % Handle a second CSV write failure.
                warning('CSV saving failed completely: %s', ME2.message); % Warn that CSV export could not be completed.
            end % End the second try block.
        else % Handle fallback-folder creation failure.
            warning('Could not create fallback folder: %s', msg); % Warn that no fallback folder is available.
        end % End the fallback-folder check.
    end % End the CSV write-protection block.
else % Handle the case where no feasible samples exist.
    warning('No CSV files were created because no feasible realization was found.'); % Warn that CSV generation was skipped.
end % End the CSV-generation condition.
