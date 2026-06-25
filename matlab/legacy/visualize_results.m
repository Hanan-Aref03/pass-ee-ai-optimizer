%% VISUALIZE_RESULTS Plot saved results for conventional or PASS mode. % Explain the visualization script purpose.

[file, path] = uigetfile('*.mat', 'Select a saved results file'); % Ask the user to select a saved MAT result file.
if isequal(file, 0) % Check whether the selection dialog was cancelled.
    error('No MAT file selected.'); % Stop because no file was selected.
end % End the file-selection check.

load(fullfile(path, file)); % Load the selected MAT result file.

isConventional = (param.rand_sc == 0); % Detect conventional mode from saved rand_sc flag.
isPASS = (param.rand_sc == 1); % Detect PASS mode from saved rand_sc flag.

if ~(isConventional || isPASS) % Reject unsupported saved modes.
    error('Unsupported mode in selected MAT file. Expected param.rand_sc=0 or 1.'); % Stop because the result file is not supported.
end % End the mode-validation condition.

if exist('valid_counts', 'var') && ~isempty(valid_counts) % Check whether feasible-sample counts were saved.
    ct_valid = valid_counts(1); % Use the saved feasible-sample count for the first QoS value.
else % Handle older saved files without valid_counts.
    ct_valid = size(loc_u_all, 3); % Fall back to the number of stored realizations.
end % End the valid-count detection.

if ct_valid == 0 % Check whether there are no feasible samples to visualize.
    error('No feasible realization exists in the selected MAT file.'); % Stop because no plot can be made.
end % End the empty-result check.

real_idx = 1; % Select the first feasible realization for geometry visualization.
Mu = param.Mu; % Store the number of users in a local variable.

%% Figure 1: Transmitter positions and users % Start the geometry figure.
figure('Name', 'Transmitter positions and users'); % Create a new figure for transmitter and user locations.
hold on; % Keep multiple scatter plots on the same axes.

if isConventional % Plot fixed BS antennas for conventional mode.
    x_vals = betax_all(:, 1, real_idx, 1); % Read fixed BS antenna x-coordinates from saved results.
    y_vals = betax_all(:, 2, real_idx, 1); % Read fixed BS antenna y-coordinates (vertical lambda/2-spaced array).
    power_vec = pbs_all(:, real_idx, 1); % Read optimized powers for the selected realization.
    marker_sz = 100 * power_vec / max(power_vec); % Scale marker size according to optimized power.
    scatter(x_vals, y_vals, marker_sz, 'filled', 'DisplayName', 'Fixed BS antennas'); % Plot fixed BS antenna positions.
    title(sprintf('Conventional fixed BS antennas, QoS R=%.2f bps/Hz', param.R)); % Add a conventional-mode title.
else % Plot optimized PASS positions.
    [Nt, Nwg, ~, ~] = size(betax_all); % Read the number of PAs and waveguides from the saved position array.
    for wg = 1:Nwg % Loop over waveguides.
        x_vals = betax_all(:, wg, real_idx, 1); % Read PA x-coordinates on the current waveguide.
        y_vals = param.betay(wg) * ones(Nt, 1); % Set all PAs on the current waveguide to the waveguide y-coordinate.
        marker_sz = 100 * pbs_all(wg, real_idx, 1) / max(pbs_all(:, real_idx, 1)); % Scale marker size using current waveguide power.
        scatter(x_vals, y_vals, marker_sz, 'filled', 'DisplayName', sprintf('WG %d (P=%.3f mW)', wg, pbs_all(wg, real_idx, 1)*1e3)); % Plot PAs on the current waveguide.
    end % End the waveguide-plot loop.
    title(sprintf('PASS (optimized) positions, QoS R=%.2f bps/Hz', param.R)); % Add a PASS-mode title.
end % End the mode-specific transmitter plot.

user_x = loc_u_all(:, 1, real_idx, 1); % Read user x-coordinates for the selected realization.
user_y = loc_u_all(:, 2, real_idx, 1); % Read user y-coordinates for the selected realization.
scatter(user_x, user_y, 120, 'r^', 'filled', 'DisplayName', 'Users'); % Plot users as red triangles.
xlabel('X (m)'); % Label the x-axis.
ylabel('Y (m)'); % Label the y-axis.
legend show; % Show the legend.
grid on; % Turn on the grid.
axis equal; % Use equal scaling on both axes.
hold off; % Release the figure axes.

%% Figure 2: Energy efficiency distribution % Start the EE histogram figure.
figure('Name', 'EE distribution'); % Create a new figure for EE distribution.
histogram(Cee_all(1:ct_valid, 1), 15, 'Normalization', 'pdf'); % Plot a normalized histogram of feasible EE values.
xlabel('Energy Efficiency (bits/Joule)'); % Label the x-axis.
ylabel('Probability Density'); % Label the y-axis.
title(sprintf('EE distribution, feasible samples=%d, QoS=%.2f bps/Hz', ct_valid, param.R)); % Add a descriptive title.
grid on; % Turn on the grid.

%% Figure 3: Sum rate vs total power % Start the rate-power tradeoff figure.
figure('Name', 'Rate vs Power'); % Create a new figure for sum-rate versus power.
sum_rate = sum(rate_all(1:ct_valid, :, 1), 2); % Compute sum rate for each feasible realization.
scatter(p_tot_all(1:ct_valid, 1), sum_rate, 60, 'b', 'filled'); % Plot total consumed power against sum rate.
xlabel('Total consumed power (W)'); % Label the x-axis.
ylabel('Sum rate (bps/Hz)'); % Label the y-axis.
title('Trade-off: sum rate vs total power'); % Add a descriptive title.
grid on; % Turn on the grid.

%% Figure 4: Boxplot of per-user rates % Start the user-rate boxplot figure.
figure('Name', 'Per-user rates'); % Create a new figure for per-user rates.
boxplot(rate_all(1:ct_valid, :, 1), 'Labels', arrayfun(@(x)sprintf('User %d', x), 1:Mu, 'UniformOutput', false)); % Draw a boxplot of rates per user.
ylabel('Rate (bps/Hz)'); % Label the y-axis.
title('Achievable rate distribution per user'); % Add a descriptive title.
grid on; % Turn on the grid.

%% Figure 5: EE across feasible realization index % Start the convergence-style EE figure.
figure('Name', 'EE across feasible realizations'); % Create a new figure for EE versus sample index.
plot(1:ct_valid, Cee_all(1:ct_valid, 1), 'b-o', 'LineWidth', 1.5); % Plot EE across feasible realization index.
xlabel('Feasible realization index'); % Label the x-axis.
ylabel('Energy Efficiency (bits/Joule)'); % Label the y-axis.
title('Energy efficiency across feasible runs'); % Add a descriptive title.
grid on; % Turn on the grid.

primary_figure_folder = fullfile(path, 'figures_from_visualize_results'); % Prefer saving figures next to the selected MAT result file.
fallback_figure_folder = fullfile(tempdir, 'PASS_figures_from_visualize_results'); % Use the system temporary folder when the MAT-file folder is not writable.
save_folder = ensureWritableFolder(primary_figure_folder, fallback_figure_folder); % Create or select a writable visualization-output folder.

figHandles = findall(0, 'Type', 'figure'); % Find all currently open figure handles.
for i = 1:length(figHandles) % Loop over all generated figures.
    figName = get(figHandles(i), 'Name'); % Read the figure name.
    safeName = regexprep(figName, '[^a-zA-Z0-9_ -]', '_'); % Replace unsafe filename characters.
    savefig(figHandles(i), fullfile(save_folder, [safeName '.fig'])); % Save the MATLAB editable FIG file.
    saveas(figHandles(i), fullfile(save_folder, [safeName '.png'])); % Save a PNG copy for quick viewing.
end % End the figure-export loop.

fprintf('Figures saved to: %s\n', save_folder); % Report the folder where figures were saved.
