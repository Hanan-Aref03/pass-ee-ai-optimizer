function betax_tx = generate_random_pass_positions(param) % Define helper that creates random PA x-positions on each waveguide.
%GENERATE_RANDOM_PASS_POSITIONS Random PA placements for fmincon_pass initial points only. % Document the helper function.
% This is NOT a simulation mode: main_pass uses rand_sc=1 with fmincon_pass optimization. % Clarify usage.
% Returns an Nt-by-Nwg matrix of PA x-positions on each waveguide. % Explain the output format.

betax_tx = zeros(param.Nt, param.Nwg); % Allocate the PA x-position matrix with size Nt by Nwg.
maxLocalAttempts = 1000; % Limit local retries for each waveguide to avoid an infinite loop.

for wg = 1:param.Nwg % Loop over each waveguide.
    placed = false; % Track whether a valid set of PA positions has been generated for this waveguide.

    for attempt = 1:maxLocalAttempts % Try multiple random placements until spacing is satisfied.
        candidate = sort(-param.D/2 + param.D * rand(param.Nt, 1)); % Generate and sort random PA x-positions within [-D/2,D/2].

        if param.Nt == 1 || all(diff(candidate) >= param.spacing) % Accept the candidate when adjacent PAs respect the spacing constraint.
            betax_tx(:, wg) = candidate; % Store the accepted candidate positions for the current waveguide.
            placed = true; % Mark the current waveguide as successfully placed.
            break; % Exit the local-attempt loop for this waveguide.
        end % End the spacing-feasibility test.
    end % End the local random-placement attempts.

    if ~placed % Check whether no valid placement was found after all attempts.
        error('Could not generate PASS initial positions satisfying the spacing constraint.'); % Stop because the geometry constraints were not satisfied.
    end % End the placement-failure check.
end % End the waveguide loop.
end % End the function.
