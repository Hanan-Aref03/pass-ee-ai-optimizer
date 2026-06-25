function [Val, betax_tx, p, r, p_total_s] = fmincon_fpa(param) % Define the conventional fixed-position antenna optimizer.
%FMINCON_FPA Optimize only transmit powers for fixed BS antennas. % Document the function objective.

if isfield(param, 'Nwg') % Check whether the shared antenna/waveguide count is defined.
    param.Na = param.Nwg; % Conventional BS has Na fixed antennas (e.g. 3); no waveguides—Nwg is only the count for comparison with PASS.
else % Handle runs that do not define Nwg.
    if ~isfield(param, 'Na') % Check whether the number of fixed antennas was provided.
        param.Na = param.Mu; % Default to one fixed antenna per user when Na is missing.
    end % End the default-Na assignment.
end % End the Na-from-Nwg assignment.

if param.Na < param.Mu % Check whether there are enough antennas to serve all users with the current association rule.
    error('Conventional mode requires param.Na >= param.Mu (set param.Nwg >= param.Mu).'); % Stop because each user needs one associated fixed antenna.
end % End the antenna-count check.

x_offsets = zeros(param.Na, 1); % Place all fixed BS antennas on x=0 (no waveguide feeds in conventional mode).
y_offsets = ((1:param.Na).' - (param.Na + 1)/2) * param.spacing; % Vertical ULA along y with lambda/2 spacing (3 antennas when Na=3).
betax_tx = [x_offsets, y_offsets]; % Store fixed BS antenna coordinates as [x,y]; positions are not optimized.

lb = zeros(param.Na, 1); % Set the lower bound of every antenna power to zero.
ub = param.pbs * ones(param.Na, 1); % Set the upper bound of every antenna power to the total BS power budget.

max_iter = 10; % Set the number of random initial points used by fmincon.
opts = optimoptions('fmincon', ... % Start the fmincon option definition.
    'Display', 'iter', ... % Display iteration output so convergence can be inspected.
    'Algorithm', 'interior-point', ... % Use the interior-point algorithm for constrained nonlinear optimization.
    'MaxFunEvals', 1e6, ... % Allow a large number of objective-function evaluations.
    'MaxIter', 1e5, ... % Allow a large number of optimizer iterations.
    'TolFun', 1e-6, ... % Set the objective-function tolerance.
    'TolX', 1e-6); % Set the optimization-variable tolerance.

F = @(X) obj_ee_fpa(X, param, betax_tx); % Define the objective function using fixed antenna coordinates.
nonlcon = @(X) nonlinear_ee_fpa(X, param, betax_tx); % Define the nonlinear constraints using fixed antenna coordinates.

A = []; % No linear inequality matrix is used.
b = []; % No linear inequality vector is used.
Aeq = []; % No linear equality matrix is used.
beq = []; % No linear equality vector is used.

optimalsetting = zeros(length(lb), max_iter); % Allocate storage for successful optimized power vectors.
ee = zeros(max_iter, 1); % Allocate storage for energy-efficiency values from each initialization.

for i = 1:max_iter % Run fmincon from multiple random initial points.
    X0 = (ub - lb) .* rand(size(lb)) + lb; % Generate a random feasible initial power vector within the bounds.
    [X, ~, exitflag] = fmincon(F, X0, A, b, Aeq, beq, lb, ub, nonlcon, opts); % Optimize the power vector.

    if exitflag > 0 % Keep only runs that converged successfully.
        ee(i) = max(-F(X), 0); % Convert the minimized negative EE back to a nonnegative EE value.
        optimalsetting(:, i) = X; % Store the successful optimized power vector.
    end % End the successful-convergence check.
end % End the random-initialization loop.

[Val, idx] = max(ee); % Select the best energy-efficiency value and its initialization index.

if Val == 0 % Check whether no successful feasible optimization run was found.
    p = zeros(param.Na, 1); % Return a zero power vector for the infeasible case.
    r = zeros(param.Mu, 1); % Return zero rates for the infeasible case.
    p_total_s = pow_total_ee(p, param); % Compute total power for the zero-power fallback.
    return; % Stop because there is no feasible optimized solution.
end % End the infeasible-optimization check.

p = optimalsetting(:, idx); % Extract the best optimized power vector.
channels = channels_gen(betax_tx, param); % Compute channels for the fixed BS antenna coordinates.
r = rate_calcul(p, param, channels, betax_tx); % Compute the final user rates for the best power vector.
p_total_s = pow_total_ee(p, param); % Compute the final total consumed power.
end % End the function.
