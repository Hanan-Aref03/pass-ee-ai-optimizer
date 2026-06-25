function [C, D] = nonlinear_ee(X, param) % Define optimized-PASS nonlinear constraints.
%NONLINEAR_EE Total power, per-user QoS, and minimum PA spacing (fmincon_pass).

[beta_tx, pn] = transX_ee(X, param); % Split the optimization vector into PA positions and power values.
channels = channels_gen(beta_tx, param); % Compute channels for the decoded PASS positions.
Ra = rate_calcul(pn, param, channels, beta_tx); % Compute user rates for the decoded variables.

C = []; % Initialize the inequality-constraint vector.
C = [C; sum(pn) - param.pbs]; % Add the total-power constraint sum(pn) <= pbs.
C = [C; param.R - Ra(:)]; % Add per-user QoS constraints Ra >= R.

for k = 1:param.Nwg % Loop over waveguides for spacing constraints.
    beta_sort = sort(beta_tx(:, k)); % Sort PA x-positions on the current waveguide.
    for n = 2:param.Nt % Loop over adjacent sorted PAs.
        C = [C; param.spacing - (beta_sort(n) - beta_sort(n - 1))]; % Add minimum-spacing constraint between adjacent PAs.
    end % End the adjacent-PA loop.
end % End the waveguide-spacing loop.

D = []; % Use no equality constraints.
end % End the legacy nonlinear-constraint function.
