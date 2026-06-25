function [channels] = channels_gen(beta_tx, param) % Define the channel-generation function for both supported modes.
%CHANNELS_GEN Generate free-space channels with no effective-index parameter. % Document that the waveguide effective index is intentionally removed.

isConventional = (param.rand_sc == 0); % Detect conventional fixed-BS-antenna mode.
isPASS = (param.rand_sc == 1); % Detect PASS mode from rand_sc alone.

if ~(isConventional || isPASS) % Reject unsupported rand_sc values.
    error('Unsupported mode. Use param.rand_sc=0 (conventional) or param.rand_sc=1 (PASS).'); % Stop with a clear mode-selection error.
end % End the mode-validation condition.

ple = sqrt(param.eta); % Compute the square-root path-gain coefficient used in the LoS channel.

if isConventional % Use the fixed-BS-antenna channel model.
    numAnt = size(beta_tx, 1); % Read the number of fixed BS antennas from the coordinate matrix.

    ant_x = reshape(beta_tx(:, 1), [numAnt 1]); % Convert antenna x-coordinates into a column vector.
    ant_y = reshape(beta_tx(:, 2), [numAnt 1]); % Convert antenna y-coordinates into a column vector.

    user_x = reshape(param.loc_u(:, 1), [1 param.Mu]); % Convert user x-coordinates into a row vector for broadcasting.
    user_y = reshape(param.loc_u(:, 2), [1 param.Mu]); % Convert user y-coordinates into a row vector for broadcasting.

    dist.bs_u = sqrt((ant_x - user_x).^2 + (ant_y - user_y).^2 + param.H^2); % Compute 3D distance from each BS antenna to each user.

    channels.bs_u = ple ./ dist.bs_u .* exp(-1j * 2*pi * dist.bs_u / param.lambda); % Compute free-space LoS channels from antennas to users.

    channels.w_tx = ones(numAnt, 1); % Store a unit placeholder because conventional mode has no waveguide.
    channels.dist = dist; % Store the distance structure for diagnostics and visualization.
    return; % Return because the conventional channel computation is complete.
end % End the conventional-mode channel branch.

beta_tx_3d = repmat(beta_tx, 1, 1, param.Mu); % Replicate PA x-positions across the user dimension.

betay_row = param.betay(:).'; % Convert waveguide y-coordinates into a row vector.
betay_3d = repmat(betay_row, param.Nt, 1, param.Mu); % Replicate waveguide y-coordinates across PAs and users.

loc_u_x = reshape(param.loc_u(:, 1), [1 1 param.Mu]); % Reshape user x-coordinates for 3D broadcasting.
loc_u_y = reshape(param.loc_u(:, 2), [1 1 param.Mu]); % Reshape user y-coordinates for 3D broadcasting.

dist.all_tx_u = sqrt((beta_tx_3d - loc_u_x).^2 + (betay_3d - loc_u_y).^2 + param.H^2); % Compute 3D distance from every PA to every user.

loc0x_mat = repmat(param.loc0(:, 1).', param.Nt, 1); % Build a matrix containing the feed-point x-coordinate of each waveguide.
dist.w_tx = abs(beta_tx - loc0x_mat); % Compute the propagation distance from the waveguide feed to each PA.

channels.tx_u = ple ./ dist.all_tx_u .* exp(-1j * 2*pi * dist.all_tx_u / param.lambda); % Compute free-space LoS channels from PAs to users.
channels.w_tx = exp(-1j * 2*pi * dist.w_tx / param.lambda); % Apply only waveguide phase rotation with no waveguide path-loss and no effective-index term.
channels.dist = dist; % Store all distance values for diagnostics and visualization.
end % End the channel-generation function.
