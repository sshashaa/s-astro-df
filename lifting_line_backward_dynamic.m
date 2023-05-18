function [ grads ] = lifting_line_backward_dynamic(turbine, atm, params, cache);
% Implement the backward propagation of the lifting line model
% Inputs: domain dictionary,
% Outputs: power per turbine (P), cache required for backprop (cache)
% See: 
% Relevant papers:
% 1) Howland, Michael F., Sanjiva K. Lele, and John O. Dabiri. 
% "Wind farm power optimization through wake steering." 
% Proceedings of the National Academy of Sciences 
% 116.29 (2019): 14495-14500.
% 2) Howland, M. F., Ghate, A. S., Lele, S. K., and Dabiri, J. O.: 
% Optimal closed-loop wake steering, Part 1: Conventionally neutral 
% atmospheric boundary layer conditions, 
% Wind Energ. Sci., https://doi.org/10.5194/wes-2020-52, 2020. 
% 3) Howland, Michael F., and John O. Dabiri. "Influence of wake model 
% superposition and secondary steering on model-based wake steering control 
% with SCADA data assimilation." Energies 14, no. 1 (2020): 52.
% 4) Howland, Michael F., et al. "Collective wind farm operation based on a 
% predictive model increases utility-scale energy production." 
% arXiv preprint arXiv:2202.06683 (2022).

%%% Initialize
% Unpack from dictionaries
% Turbine info
Nt = turbine.Nt; turbCenter = turbine.turbCenter;
isFront = turbine.isFront; isBack = turbine.isBack; %graph
% leadingTurbine = turbine.leadingTurbine; 
% followingTurbine = turbine.followingTurbine;
D = turbine.D; D = ones(Nt,1) * D;
A = D.^2 * pi/4;
D = D ./ D; R = D/2; % by convention D=1 
yaw = turbine.yaw; %* pi / 180; % convert to radians
% Atmospheric conditions
rho = atm.rho; 
% Model parameters
kw = params.kw; sigma_0 = params.sigma_0;
delta_w = R; eps = params.eps;
% Initialize variables in function
grads = {};
du_eff_dyc = zeros(Nt,1); du_eff_da = zeros(Nt,1); dy_c_dgamma = zeros(Nt,1); 
da_dgamma = zeros(Nt,1); dcp_dap = zeros(Nt,1); dp_du_eff = zeros(Nt,1);
dp_dcp = zeros(Nt,1); dcp_dgamma = zeros(Nt,1);
% Get stuff from cache
Cp = cache.Cp; u_eff = cache.u_eff; a = cache.a; xp = cache.xp;
yCenter = cache.yCenter; sigmaEnd = cache.sigmaEnd; uInf = cache.uInf;
ap = cache.ap; duSquareSum = cache.duSquareSum;
deltaUIndividual = cache.deltaUIndividual; 
gaussianStore = cache.gaussianStore;
%%%

%% Backprop
% Loop over turbines
dp_dgamma = zeros(Nt,1);
for i = Nt:-1:1;
    % Same for all turbines
    % d a / d gamma
    da_dgamma(i) = -0.5*cache.Ct(i)*sin(yaw(i))*cos(yaw(i))/sqrt(1-cache.Ct(i)*cos(yaw(i))^2 + eps);
    % dP / d u_eff
    dp_du_eff(i) = 0.5*rho*A(i)*Cp(i)*u_eff(i)^2 * 3;
    % dP / dcp
    dp_dcp(i) = 0.5*rho*A(i)*u_eff(i)^3;
    % d cp / da
    %dcp_dap(i) = -4*eta*(1 - 2*ap(i) + ap(i)^2);
    dcp_dap(i) = 4*cache.eta(i)*(1 - 4*ap(i) + 3*ap(i)^2) * cos(yaw(i))^2;
    dcp_dgamma(i) = -4 * cache.eta(i) * ap(i) * (1-ap(i))^2 * sin(yaw(i)) * cos(yaw(i))^(params.powerExp-1) * params.powerExp;
    dp_dgamma(i) = dp_dgamma(i) + dp_dcp(i)*dcp_dgamma(i);
end
for i = Nt:-1:1;
    turbine.turbinesInBack{i} = [];
    for j = 1:Nt;
        dx = turbCenter(j, 1) - turbCenter(i, 1);
        if dx > 0;
           dw = 1 + kw(i) * log(1 + exp((dx - 2.0 .* delta_w(i)) / R(i)));
           boundLow = turbCenter(i,2)-dw/2;
           boundHigh = turbCenter(i,2)+dw/2;
           edgeLow = turbCenter(j,2)-D(j)/2;
           edgeHigh = turbCenter(j,2)+D(j)/2;
           if (edgeLow>=boundLow & edgeLow<=boundHigh) || ...
                   (edgeHigh>=boundLow & edgeHigh<=boundHigh);
              turbine.turbinesInBack{i} = [turbine.turbinesInBack{i}, j]; 
           end
        end
    end
    % dP / dgamma
    %dp_dgamma(i) = dp_dgamma(i) + dp_dcp(i)*dcp_da(i)*da_dgamma(j);
    % Flip over whether the turbine sees a wake or not
    %j = followingTurbine(i);
    du_eff_da = 0;
    for k = turbine.turbinesInBack{i};
        if strcmp(params.superposition, 'sos');
            du_eff_da = -(duSquareSum(k) + params.eps)^(-0.5) ...
                * deltaUIndividual{k}(i) * cache.delta_u_face_store(i, k) / (a(i)+eps);
        else
            du_eff_da = -deltaUIndividual{k}(i) / (a(i)+eps) * gaussianStore(i, k);
        end
        dp_dgamma(i) = dp_dgamma(i) + dp_du_eff(k)*du_eff_da*da_dgamma(i);
        % d u_eff / d y_c
        du_eff_dyc(i) = -(1/D(k)) * (cache.delta_u_face_store(k) * D(k)^2 / (8*sigma_0(i)^2) * ...
                (-exp(-(turbCenter(k, 2)+D(k)/2-yCenter(i, k))^2/(2*sigmaEnd(i, k)^2)) + ...
                exp(-(turbCenter(k, 2)-D(k)/2-yCenter(i, k))^2/(2*sigmaEnd(i, k)^2))));
        %%%
        % d y_c / d gamma
        dw = 1 + kw(i) * log(1 + exp((xp{i, k} - 2.0 .* delta_w(i)) / R(i)));
        % Changed this when using the local velocity to compute y_c
        dy_c_dgamma(i) = (cos(yaw(i))^3-2*sin(yaw(i))^2*cos(yaw(i)))*...
                trapz(xp{i, k}, -0.25*cache.Ct(i)*0.5*(1+erf(xp{i, k}./(delta_w(k)*sqrt(2))))./dw.^2);
        % Sum together to get dP / dgamma
        dp_dgamma(i) = dp_dgamma(i) + dp_du_eff(k)*du_eff_dyc(i)*dy_c_dgamma(i); 
        % Extra terms from secondary steering
        if strcmp(params.superposition, 'mom') == true && params.secondary==true;
            for m = cache.turbinesInFront{i};
                % d y_c / d gamma
                dy_c_dgamma(m) = (u_eff(m)/u_eff(i)) * cache.ucr(m, i) ...
                    * (cos(yaw(m))^3-2*sin(yaw(m))^2*cos(yaw(m)))*...
                        trapz(xp{i, k}, -0.25*cache.Ct(m)*0.5*(1+erf(xp{i, k}./(delta_w(k)*sqrt(2))))./dw.^2);
                % Sum together to get dP / dgamma
                dp_dgamma(m) = dp_dgamma(m) + dp_du_eff(k)*du_eff_dyc(i)*dy_c_dgamma(m); 
            end
        end
    end
end
% Extra terms from convective superposition
if strcmp(params.superposition, 'mom') == true;
    for i = Nt:-1:1;
        for m = cache.turbinesInFront{i};
            for k = turbine.turbinesInBack{i};
                du_eff_dueffUp = -cache.erfStore(i, k);
                % 1
                du_eff_da = -deltaUIndividual{i}(m) / (a(m)+eps) * gaussianStore(m, i);
                dp_dgamma(m) = dp_dgamma(m) + ...
                    dp_du_eff(k)*du_eff_dueffUp*du_eff_da*da_dgamma(m); 
                % 2
                % d u_eff / d y_c
                du_eff_dyc(m) = -(1/D(i)) * (cache.delta_u_face_store(i) * D(i)^2 / (8*sigma_0(m)^2) * ...
                        (-exp(-(turbCenter(i, 2)+D(i)/2-yCenter(m, i))^2/(2*sigmaEnd(m, i)^2)) + ...
                        exp(-(turbCenter(i, 2)-D(i)/2-yCenter(m, i))^2/(2*sigmaEnd(m, i)^2))));
                % d y_c / d gamma
                dw = 1 + kw(m) * log(1 + exp((xp{m, i} - 2.0 .* delta_w(m)) / R(m)));
                dy_c_dgamma(m) = (cos(yaw(m))^3-2*sin(yaw(m))^2*cos(yaw(m)))*...
                        trapz(xp{m, i}, -0.25*cache.Ct(m)*0.5*(1+erf(xp{m, i}./(delta_w(m)*sqrt(2))))./dw.^2);
                dp_dgamma(m) = dp_dgamma(m) + ...
                    dp_du_eff(k)*du_eff_dueffUp*du_eff_dyc(m)*dy_c_dgamma(m); 
            end
        end
    end
end

% Store gradients in grads for gradient checking and sensitivity
grads.da_dgamma = da_dgamma;
grads.dp_du_eff = dp_du_eff;
grads.dp_dcp = dp_dcp;
grads.dcp_dgamma = dcp_dgamma;
grads.du_eff_dyc = du_eff_dyc;
grads.du_eff_da = du_eff_da;
grads.dy_c_dgamma = dy_c_dgamma;
grads.dp_dgamma = dp_dgamma;

% Derivative of money w.r.t. power
dm_dp = params.price * params.time / 10^6; % Convert to MW

grads.dm_dgamma = dp_dgamma * dm_dp;

end

