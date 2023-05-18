function [ turbine, atm, params ] = initalize_inputs(X)
%This function initializes the parameters required by the model

% Initialize the structure variables
turbine = {}; atm = {}; params = {};

turbine.Nt = length(X(:,1));
% Define with respect to zero degrees inflow
turbine.turbCenter = X; X_init = X;
[val, ind] = sort(X(:,1));
X = X(ind,:);
turbine.turbCenter = X; X_store = X;
indSorted = ind;

% Lifting line model parameters
turbine.D = 90; params.ratedPower = 3000;
% Outdated
turbine.isFront = zeros(1, turbine.Nt); 
turbine.isBack = zeros(1, turbine.Nt);
turbine.turbine_type = {};
for i=1:turbine.Nt;
    turbine.turbine_type{i} = 'XXX';
end

% Yaw initialization
turbine.yaw = zeros(turbine.Nt,1); % just the baseline
mBaseTotal = 0;
atm.wind_speed = ones(turbine.Nt,1) * 7.5;
atm.rho = 1.225;

% Wake model parameters
params.kw = ones(turbine.Nt,1)*0.1; 
params.sigma_0 = ones(turbine.Nt,1) * 0.5;
params.liftingLinePower = false; 
params.eps = 10^-8; % for numerical stability
params.Nx = 500;
params.superposition = 'mod';
params.powerExp = 2;

% Economic stuff
% Costs
beta = 5*10^(-3); theta = 1*10^(-3);
cost_turbineFull = 2*10^6; % $ per turbine
% Price of electricity
years = 15;
timeFull = years*365*24;
params.time = 0;
params.price = 75; % $/MWh
params.cost_turbine = cost_turbineFull;
scale = 100;
params.cost_linear = beta*params.cost_turbine*scale;
params.cost_quad = theta*params.cost_turbine*scale;

end

