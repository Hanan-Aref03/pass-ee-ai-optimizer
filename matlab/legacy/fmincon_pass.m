function [Val, betax_tx, p, r, p_total_s] = fmincon_pass(param) % Define the joint PASS optimizer for PA positions and powers.
%FMINCON_PASS Jointly optimize PA x-positions and waveguide powers for PASS. % Document the function objective.
% Requires param.rand_sc = 1 (PASS channel model). % Explain required mode flag.

if param.rand_sc ~= 1 % Check that the PASS channel model flag is enabled.
    error('fmincon_pass requires param.rand_sc = 1.'); % Stop because channels_gen needs rand_sc=1 for PASS physics.
end % End the rand_sc validation.

if param.Nwg < param.Mu % Check whether there are enough waveguides for the user-association rule.
    error('PASS mode requires param.Nwg >= param.Mu.'); % Stop because each user needs one associated waveguide.
end % End the waveguide-count check.

nPos = param.Nt * param.Nwg; % Count the number of optimized PA x-position variables.
nPow = param.Nwg; % Count the number of optimized waveguide power variables.
nVar = nPos + nPow; % Count the total number of optimization variables.

lb = [-param.D/2 * ones(nPos, 1); zeros(nPow, 1)]; % Lower-bound PA positions inside the area and powers at zero.
ub = [param.D/2 * ones(nPos, 1); param.pbs * ones(nPow, 1)]; % Upper-bound PA positions and per-waveguide power by the BS budget.

if isfield(param, 'fmincon_fast') && param.fmincon_fast % Use faster solver settings when building large ML datasets.
    max_iter = 3; % Use fewer random starts to reduce runtime per sample.
    opts = optimoptions('fmincon', ... % Start the fast fmincon option definition.
        'Display', 'off', ... % Suppress iteration printing during batch dataset generation.
        'Algorithm', 'interior-point', ... % Use the interior-point algorithm for constrained nonlinear optimization.
        'MaxFunEvals', 5e4, ... % Allow fewer objective-function evaluations in fast mode.
        'MaxIter', 500, ... % Allow fewer optimizer iterations in fast mode.
        'TolFun', 1e-5, ... % Relax the objective-function tolerance slightly in fast mode.
        'TolX', 1e-5); % Relax the optimization-variable tolerance slightly in fast mode.
else % Use accurate solver settings for thesis runs and defense plots.
    max_iter = 10; % Set the number of random initial points used by fmincon.
    opts = optimoptions('fmincon', ... % Start the standard fmincon option definition.
        'Display', 'iter', ... % Display iteration output so convergence can be inspected.
        'Algorithm', 'interior-point', ... % Use the interior-point algorithm for constrained nonlinear optimization.
        'MaxFunEvals', 1e6, ... % Allow a large number of objective-function evaluations.
        'MaxIter', 1e5, ... % Allow a large number of optimizer iterations.
        'TolFun', 1e-6, ... % Set the objective-function tolerance.
        'TolX', 1e-6); % Set the optimization-variable tolerance.
end % End the fast-vs-standard solver selection.

if isfield(param, 'fmincon_restarts') && ~isempty(param.fmincon_restarts) % Allow manual override of the restart count.
    max_iter = param.fmincon_restarts; % Replace the restart count when the caller sets fmincon_restarts.
end % End the optional restart override.

F = @(X) obj_ee(X, param); % Define the negative-EE objective for joint position and power optimization.
nonlcon = @(X) nonlinear_ee(X, param); % Define total-power, QoS, and spacing constraints.

A = []; % No linear inequality matrix is used.
b = []; % No linear inequality vector is used.
Aeq = []; % No linear equality matrix is used.
beq = []; % No linear equality vector is used.

optimalsetting = zeros(nVar, max_iter); % Allocate storage for successful optimized solution vectors.
ee = zeros(max_iter, 1); % Allocate storage for energy-efficiency values from each initialization.

for i = 1:max_iter % Run fmincon from multiple random initial points.
    betax0 = generate_random_pass_positions(param); % Generate a spacing-feasible random PA geometry for initialization.
    p0 = (ub(nPos + 1:end) - lb(nPos + 1:end)) .* rand(nPow, 1) + lb(nPos + 1:end); % Generate a random initial power vector within bounds.
    X0 = [betax0(:); p0]; % Stack positions and powers into one optimization vector.

    [X, ~, exitflag] = fmincon(F, X0, A, b, Aeq, beq, lb, ub, nonlcon, opts); % Optimize PA positions and waveguide powers.

    if exitflag > 0 % Keep only runs that converged successfully.
        ee(i) = max(-F(X), 0); % Convert the minimized negative EE back to a nonnegative EE value.
        optimalsetting(:, i) = X; % Store the successful optimized solution vector.
    end % End the successful-convergence check.
end % End the random-initialization loop.

[Val, idx] = max(ee); % Select the best energy-efficiency value and its initialization index.

if Val == 0 % Check whether no successful feasible optimization run was found.
    betax_tx = zeros(param.Nt, param.Nwg); % Return a zero PA-position matrix for the infeasible case.
    p = zeros(param.Nwg, 1); % Return a zero power vector for the infeasible case.
    r = zeros(param.Mu, 1); % Return zero rates for the infeasible case.
    p_total_s = pow_total_ee(p, param); % Compute total power for the zero-power fallback.
    return; % Stop because there is no feasible optimized solution.
end % End the infeasible-optimization check.

Xbest = optimalsetting(:, idx); % Extract the best optimized solution vector.
[betax_tx, p] = transX_ee(Xbest, param); % Decode optimized positions and powers from the solution vector.
channels = channels_gen(betax_tx, param); % Compute channels for the optimized PASS geometry.
r = rate_calcul(p, param, channels, betax_tx); % Compute the final user rates for the best solution.
p_total_s = pow_total_ee(p, param); % Compute the final total consumed power.
end % End the function.
