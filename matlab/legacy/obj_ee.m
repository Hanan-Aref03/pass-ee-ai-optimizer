function [obj] = obj_ee(X, param) % Define the optimized-PASS objective function.
%OBJ_EE Return negative EE for joint PA-position and power optimization (fmincon_pass).

[beta_tx, p] = transX_ee(X, param); % Split the optimization vector into PA positions and power values.
channels = channels_gen(beta_tx, param); % Compute channels for the decoded PASS positions.
r = rate_calcul(p, param, channels, beta_tx); % Compute user rates for the decoded variables.
p_total_s = pow_total_ee(p, param); % Compute total consumed power for the decoded power vector.

obj = -sum(r) / p_total_s; % Return negative energy efficiency for minimization-based solvers.
end % End the legacy objective function.
