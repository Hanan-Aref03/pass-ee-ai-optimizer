function [ptotal] = pow_total_ee(p, param, varargin) % Define the total consumed-power function.
%POW_TOTAL_EE Compute transmit power plus circuit power only. % Document the simplified power-consumption model.

ptotal = sum(p) + param.pcircuit; % Add total transmit power and fixed circuit power; no waveguide path-loss term is included.
end % End the power-consumption function.
