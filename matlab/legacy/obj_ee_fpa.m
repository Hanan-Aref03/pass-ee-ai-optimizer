function obj = obj_ee_fpa(X, param, betax_tx) % Define the energy-efficiency objective for conventional mode.
%OBJ_EE_FPA Return negative EE so fmincon can maximize EE by minimization. % Document the objective convention.

p = X(:); % Convert the optimization variable into a column power vector.
channels = channels_gen(betax_tx, param); % Compute channels from fixed BS antennas to users.
r = rate_calcul(p, param, channels, betax_tx); % Compute achievable rates for the current power vector.
p_total = pow_total_ee(p, param); % Compute total consumed power for the current power vector.

obj = -sum(r) / p_total; % Return negative energy efficiency because fmincon minimizes objectives.
end % End the objective function.
