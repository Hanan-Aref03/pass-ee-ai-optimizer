function [R] = rate_calcul(p, param, channels, beta_tx) % Define the rate-calculation function for both supported modes.
%RATE_CALCUL Compute per-user achievable rates using SINR. % Document the purpose of the function.

isConventional = (param.rand_sc == 0); % Detect conventional fixed-BS-antenna mode.
isPASS = (param.rand_sc == 1); % Detect PASS mode from rand_sc alone.

if ~(isConventional || isPASS) % Reject unsupported rand_sc values.
    error('Unsupported mode. Use param.rand_sc=0 (conventional) or param.rand_sc=1 (PASS).'); % Stop with a clear mode-selection error.
end % End the mode-validation condition.

R = zeros(param.Mu, 1); % Allocate the per-user rate vector.

if isConventional % Use the fixed-BS-antenna SINR model.
    numAnt = size(beta_tx, 1); % Read the number of fixed BS antennas.

    if param.Mu > numAnt % Ensure that each user has one associated antenna under the current model.
        error('Conventional mode requires at least one BS antenna per user. Set param.Na >= param.Mu.'); % Stop if there are fewer antennas than users.
    end % End the antenna-count check.

    for m = 1:param.Mu % Loop over each user.
        desired_power = p(m) * abs(channels.bs_u(m, m))^2; % Compute desired received power from antenna m to user m.

        interference_power = 0; % Initialize the interference power seen by user m.
        for k = 1:numAnt % Loop over all fixed BS antennas.
            if k ~= m % Exclude the desired antenna of user m.
                interference_power = interference_power + p(k) * abs(channels.bs_u(k, m))^2; % Add interference from antenna k to user m.
            end % End the desired-antenna exclusion.
        end % End the interference-antenna loop.

        gamma_sinr = desired_power / (interference_power + param.sigmanoise); % Compute SINR for user m.
        R(m, 1) = param.B * log2(1 + gamma_sinr); % Convert SINR into achievable rate in bps/Hz.
    end % End the user loop.

    return; % Return because the conventional rate computation is complete.
end % End the conventional-mode branch.

if param.Mu > param.Nwg % Ensure that each user has one associated waveguide under the current model.
    error('PASS mode requires at least one waveguide per user. Set param.Nwg >= param.Mu.'); % Stop if there are fewer waveguides than users.
end % End the waveguide-count check.

for m = 1:param.Mu % Loop over each user.
    desired_signal = 0; % Initialize the coherent desired signal received by user m.

    for n = 1:param.Nt % Loop over all PAs on the desired waveguide.
        desired_signal = desired_signal + channels.tx_u(n, m, m) * channels.w_tx(n, m); % Add the PA contribution from waveguide m to user m.
    end % End the desired-PA loop.

    desired_power = p(m) * abs(desired_signal)^2; % Compute desired received power for user m.

    interference_power = 0; % Initialize the interference power seen by user m.
    for k = 1:param.Nwg % Loop over all waveguides.
        if k ~= m % Exclude the desired waveguide of user m.
            interf_signal = 0; % Initialize the interfering coherent signal from waveguide k.
            for n = 1:param.Nt % Loop over all PAs on interfering waveguide k.
                interf_signal = interf_signal + channels.tx_u(n, k, m) * channels.w_tx(n, k); % Add the PA contribution from waveguide k to user m.
            end % End the interfering-PA loop.
            interference_power = interference_power + p(k) * abs(interf_signal)^2; % Add waveguide-k interference power to user m.
        end % End the desired-waveguide exclusion.
    end % End the interference-waveguide loop.

    gamma_sinr = desired_power / (interference_power + param.sigmanoise); % Compute SINR for user m.
    R(m, 1) = param.B * log2(1 + gamma_sinr); % Convert SINR into achievable rate in bps/Hz.
end % End the user loop.
end % End the rate-calculation function.
