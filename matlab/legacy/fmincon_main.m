function [Val, betax_tx, p, r, p_total_s] = fmincon_main(param) % Define the entry-point function that runs the selected transmission mode.
%FMINCON_MAIN Run one of two modes selected by param.rand_sc only. % Document the purpose of this file.
% rand_sc=0 -> conventional: 3 fixed BS antennas (Na=Nwg), power optimized via fmincon_fpa. % Explain conventional mode.
% rand_sc=1 -> PASS: 3 waveguides with Nt PAs each, joint position+power optimization via fmincon_pass. % Explain PASS mode.

if param.rand_sc == 0 % Check whether the requested mode is conventional fixed BS.
    [Val, betax_tx, p, r, p_total_s] = fmincon_fpa(param); % Optimize transmit powers only for fixed antenna positions.
    return; % Stop this function after conventional-mode processing is complete.
end % End the conventional-mode branch.

if param.rand_sc == 1 % Check whether the requested mode is PASS.
    [Val, betax_tx, p, r, p_total_s] = fmincon_pass(param); % Jointly optimize PA x-positions and waveguide powers.
    return; % Stop this function after PASS processing is complete.
end % End the PASS-mode branch.

error('Unsupported mode. Use param.rand_sc=0 (conventional) or param.rand_sc=1 (PASS).'); % Stop for invalid rand_sc values.
end % End the main function.
