function r = extrapVentRadius(Q)
% Volcanic conduit radius extrapolation for control scenarios. 
% Based on an emprical curve fit for conduit area derived from conduit 
% model simulations featured in:
%       Rowell et al. (2022), Frontiers in Earth Science
% 
% Q = total mass discharge, in kg/s


    % Emprical power law curve fit for radius as f of q
%     rfun = @(q) 6.059e-06.*log10(q).^7.559 + 5.073;
%     rfun = @(q) 6.084e-06.*log10(q).^7.556 + 5.072;
    
    % Empirical curve fit for conduit area
    C1 = 0.009913;
    C2 = 0.7267;
    C3 = 58.04;
    afun = @(x) C1.*(x).^C2 + C3;

%     r = rfun(Q);
    r = (afun(Q)./pi).^(1/2);
    
end