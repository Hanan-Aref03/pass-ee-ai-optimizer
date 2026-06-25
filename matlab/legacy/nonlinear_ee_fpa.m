function [C, Ceq] = nonlinear_ee_fpa(X, param, betax_tx) % Define nonlinear constraints for conventional mode.
%NONLINEAR_EE_FPA Enforce total power and per-user QoS constraints. % Document the constraint purpose.

p = X(:); % Convert the optimization variable into a column power vector.

C1 = sum(p) - param.pbs; % Build the total-power inequality sum(p) <= pbs as C1 <= 0.

channels = channels_gen(betax_tx, param); % Compute channels for the fixed BS antenna coordinates.
r = rate_calcul(p, param, channels, betax_tx); % Compute user rates for the current power vector.

C2 = param.R - r(:); % Build the QoS inequalities r >= R as R-r <= 0.

C = [C1; C2]; % Stack all inequality constraints into one vector.
Ceq = []; % Use no equality constraints.
end % End the nonlinear-constraint function.
