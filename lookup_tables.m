function [ powerOut, CpOut, CtOut, dCt_dyaw, eta ] = lookup_tables(ws, A, yaw, atm, two);
%Experimental Ct and Cp look up tables
% Parameters
rho = atm.rho;


%%
    currData = readmatrix('Wake Effect Dataset(Pair1).csv');
    % limit to below rated wind speed
    rWS = 15;
    %% Spline model for power curve
    quant_range_basis = quantile(currData(:,4),[0.3333333333 0.6666666667]);
    % limit to below rated wind speed
    udataPC = max(currData(:,2),currData(:,3));
    % generate the basic spline model
    ws_basis_mat = WindSpeedBasisReg(udataPC(:,1), quant_range_basis);
    % regression using splines
    p = max(currData(:,6),currData(:,7));
    sp = fitlm(ws_basis_mat,p);

%     p_pred = spPredict(udataPC, sp, quant_range_basis);
%     p_pred(p_pred > 100) = 100;
% powerOut = spPredict (ws, sp, quant_range_basis);


%%
% 5-25 m/s
% wind_speed = linspace(3, 20, 20-2);
% wind_speed_ct = wind_speed;
% 
% % Power curve
% poly_1 = 1.0e+04 * [0.303351137777127  -4.218819053193799];
% poly_2 = 1.0e+06 * [0.000616236162362   1.459763837638376];
% p_model = poly_1(1)*wind_speed.^3 + poly_1(2);
% ind=find(wind_speed>=9 & wind_speed<11);
% p_model(ind) = poly_2(1)*wind_speed(ind).^3 + poly_2(2);
% ind=find(wind_speed>=11);
% p_model(ind)=2100*1000;
% power = p_model;
% 
% % Thrust curve
% Ct_model = zeros(length(wind_speed),1);
% Ct_model(wind_speed<8) = 0.83;
% ind = find(wind_speed>=8);
% poly = [-0.000714160839161   0.037802947052947  -0.675690934065938   4.160945804195820];
% Ct_model(ind) = polyval(poly,wind_speed(ind));
% Ct = Ct_model;
% 

    dat = readmatrix("Ct_V90.csv");
    dat_ws = dat(:,1);
    dat_Ct = dat(:,2);
    Ct_fit=fit(dat_ws,dat_Ct,'smoothingspline');
    
% % Out values
if ws >= 3 && ws <= 15
    powerOut = spPredict (ws, sp, quant_range_basis);
    if powerOut > 100
        powerOut = 100;
    end
    CpOut = powerOut / (0.5*rho*A*ws^3);
    CtOut  = Ct_fit(ws);
    CtOut = min(CtOut,1); 
    ap = 0.5*(1-sqrt(1-CtOut));
    eta = CpOut / (4*ap*(1-ap)^2);
elseif ws > 15
    powerOut = 100;
    CpOut = powerOut / (0.5*rho*A*ws^3);
    CtOut  = Ct_fit(ws);
    CtOut = min(CtOut,1); 
    ap = 0.5*(1-sqrt(1-CtOut));
    eta = CpOut / (4*ap*(1-ap)^2);
elseif ws < wind_speed(1) % below or above the cut-in speed
    powerOut = 0;
    CpOut = 0; 
    CtOut = 0; 
    eta = 0;
end

dCt_dyaw = 0;
% 
%% 
function WindSpeedBasisMatrix = WindSpeedBasisReg(x, quant_range_basis)

% generate basis matrix for wind speeds
b0 = zeros(length(x),8);
b1 = zeros(length(x),7);
b2 = zeros(length(x),6);
b3 = zeros(length(x),5);
% quant_range_basis = quantile(currData(:,1),[0.3333333333 0.6666666667]);
kn = [0 0 0 quant_range_basis(1) quant_range_basis(2) 15 15 15 15];

for j = 1:length(x)
for i = 1:8
    % 0th order
    if x(j)>=kn(i) && x(j)<kn(i+1)
        b0(j,i) = 1;
    end
end
for i = 1:7
    % 1st order
    if b0(j,i) == 0
        b1(j,i) = (kn(i+2) - x(j))/(kn(i+2) - kn(i+1))*b0(j,i+1);
    elseif b0(j,i+1) == 0
        b1(j,i) = (x(j) - kn(i))/(kn(i+1) - kn(i))*b0(j,i);
    else
        b1(j,i) = (x(j) - kn(i))/(kn(i+1) - kn(i))*b0(j,i) + (kn(i+2) - x(j))/(kn(i+2) - kn(i+1))*b0(j,i+1);
    end
    if isnan(b1(j,i))
        b1(j,i) = 0;
    end
end
% b1(isnan(b1)) = 0;
for i = 1:6
    % 2nd order
    if b1(j,i) == 0
        b2(j,i) = (kn(i+3) - x(j))/(kn(i+3) - kn(i+1))*b1(j,i+1);
    elseif b1(j,i+1) == 0
        b2(j,i) = (x(j) - kn(i))/(kn(i+2) - kn(i))*b1(j,i);
    else
        b2(j,i) = (x(j) - kn(i))/(kn(i+2) - kn(i))*b1(j,i) + (kn(i+3) - x(j))/(kn(i+3) - kn(i+1))*b1(j,i+1);
    end
    if isnan(b2(j,i))
        b2(j,i) = 0;
    end
end
% b2(isnan(b2)) = 0;
for i = 1:5
    % 3rd order
    if b2(j,i) == 0
        b3(j,i) = (kn(i+4) - x(j))/(kn(i+4) - kn(i+1))*b2(j,i+1);
    elseif b2(j,i+1) == 0
        b3(j,i) = (x(j) - kn(i))/(kn(i+3) - kn(i))*b2(j,i);
    else
        b3(j,i) = (x(j) - kn(i))/(kn(i+3) - kn(i))*b2(j,i) + (kn(i+4) - x(j))/(kn(i+4) - kn(i+1))*b2(j,i+1);
    end
    if isnan(b3(j,i))
        b3(j,i) = 0;
    end
end
% b3(isnan(b3)) = 0;
end
WindSpeedBasisMatrix = b3;
end

function yp = spPredict(xp, sp, quant_range_basis)
% predict power using regression and splines 
% wind speeds are generated using wake model
yp = xp;
for u = 1:size(xp,2)
    w = WindSpeedBasisReg(xp(:,u), quant_range_basis);
    yp(:,u) = predict(sp,w);
    if yp(:,u) > 100
        yp(:,u) = 100;
    end
end
yp(xp>=14.5) = 100;
end

end

