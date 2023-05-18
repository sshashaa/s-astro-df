%% Post process the static wake steering experiment results
function P_final = Howland_model(kw_pt,sig0_pt,direction_pt,offset_pt,ws)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Michael F. Howland, mhowland@mit.edu
% This code is a companion to Howland et al. "Collective wind farm
% operation based on a predictive model increases utility-scale energy 
% production" 
% This script will produce a similar output to Figure 4 in the manuscript,
% although we note that the flow control model predictions here will not be 
% identical to the predictions in the paper because the coefficients of 
% lift, drag, thrust, and power are not available publicly.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add paths
addpath(strcat('utility_functions\figplots'))
addpath(strcat('utility_functions'))
addpath(strcat('flow_control_model'))

goldenrod = [0.85, 0.65, 0.13];

%% Settings
plotNCut = 25; % minimum number of data points to plot a wind direction bin

% Bootstrapping
dataBoot = {};
dataBoot.conf = 0.95; % percent confidence interval
dataBoot.Boots = 1000;

% Wind farm data
X = [0 0; 3.4*cosd(307.1) 3.4*sind(307.1)];



% Load data
TOI = [1, 2];
turbines = {'01','02'};
rotateTurb = 1; X = X - X(rotateTurb,:); X = X(TOI,:);
% Wind condition bins
wind_speed_center = ws; % m/s
% Load wake model data

% Angle of alignment
alignment_angle = 307.1;



% Initialize wake model
[ turbine, atm, params ] = initalize_inputs(X);
% Wake model setup
params.superposition = 'mod'; 
params.secondary = true;
params_opti.var_p=0.01; params_opti.var_k=5E-5; params_opti.var_sig=5E-5; 
params.Prs = sqrt(params_opti.var_p)*ones(turbine.Nt,1);
params.ucMaxIt = 10; params.Ny = 100; params.epsUc = 10^(-3);
params.yaw_init = zeros(turbine.Nt,1);

%% Power-yaw model inputs
% Below, input the turbine specific airfoil parameters
% The airfoil parameters for the turbine of interest in this experiment are
% not available publicly
airfoil_params = {};
% airfoil_params.R_twist = out.R; airfoil_params.twist = out.twist*pi/180; airfoil_params.c = out.c;
% airfoil_params.interp_setting = out.interp_setting; airfoil_params.airfoil = out.airfoil;
% airfoil_params.interp_pairs = out.interp_pairs; airfoil_params.thickness = out.thickness;
% airfoil_params.tilt = INSERT; airfoil_params.pitch_struct = INSERT;
% airfoil_params.interp_thick = out.interp_thick;
% airfoil_params.tsr = INSERT;
% airfoil_params.zh = INSERT;
% Model settings
params.semi_empirical = false;
params.cosine_model = [true, true];
params.local_speed = true;
% Compare to cosine models of the past
params.powerExp=2;

% Initialize loop
% Cosine model
% Loop over all wind directions
    [ XR, indSorted, unsort ] = rotate( X, direction_pt, rotateTurb );
    params.kw = kw_pt*ones(4,1); params.sigma_0 = sig0_pt*ones(4,1);
    turbine.turbCenter = XR; 
    atm.wind_speed = wind_speed_center*ones(turbine.Nt,1); % input real wind speed
            % Lidar
            ABL_data = {};
%             ABL_data.uv = uv(:, local_ind_wind); 
%             ABL_data.alpha = alpha(:, local_ind_wind);
%             ABL_data.heights = heights;
            ABL_data.speed_ratio = 0; % INSERT
                % Turbine
                turbine_data = {};
                turbine_data.zhub = 90;
                % Turbine specific details for power-yaw model
                % Yawed turbine
                turbine_data.pitch_yawed = 0; % INSERT
                turbine_data.gamma_set = 0; % INSERT
                turbine_data.lambda_yawed = 0; % INSERT
                turbine_data.WFC_strategy = 0; % INSERT
                % Baseline turbine
                turbine_data.pitch_base = 0; % INSERT
                turbine_data.lambda_base = 0; % INSERT
                turbine_data.gamma_base = 0; % INSERT
                % Store in dictionaries
                turbine.turbine_data = turbine_data;
                turbine.airfoil_params = airfoil_params;
                atm.ABL_data = ABL_data;
                % Run forward model
                turbine.yaw = zeros(turbine.Nt,1);
                turbine.D = 80;
%                 turbine.yaw(1) = turbine_1_yaw(local_ind)*pi/180;
                turbine.yaw = turbine.yaw(indSorted);
                [ P, cache, ~ ] = lifting_line_forward_dynamic(turbine, atm, params);
                P = P(unsort); 

       
P_final = P';

