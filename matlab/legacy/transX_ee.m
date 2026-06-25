function [beta_tx, pn] = transX_ee(X, param) % Define the legacy optimization-vector decoder.
%TRANSX_EE Convert a vector into PA x-positions and waveguide powers. % Document the helper function.

idx = 1; % Initialize the reading index inside the optimization vector.
beta_tx = zeros(param.Nt, param.Nwg); % Allocate the PA x-position matrix.
pn = zeros(param.Nwg, 1); % Allocate the waveguide-power vector.

for k = 1:param.Nwg % Loop over waveguides.
    for n = 1:param.Nt % Loop over PAs on each waveguide.
        beta_tx(n, k) = X(idx); % Read one PA x-position from the optimization vector.
        idx = idx + 1; % Move to the next optimization-vector element.
    end % End the PA loop.
end % End the waveguide loop for PA positions.

for k = 1:param.Nwg % Loop over waveguides for power values.
    pn(k, 1) = X(idx); % Read one waveguide power from the optimization vector.
    idx = idx + 1; % Move to the next optimization-vector element.
end % End the power-reading loop.
end % End the decoder function.
