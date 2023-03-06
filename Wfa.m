function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov, r_ws] = Wfa(theta, runlength, problemRng, seed, m_rep)

FnGrad = NaN;
FnGradCov = NaN;
constraint = NaN;
ConstraintCov = NaN;
ConstraintGrad = NaN;
ConstraintGradCov = NaN;
r_ws = NaN;

% Initialize the variables at the start of first function call 
persistent currData coordinate turbineDia x1 fPower rWS Ct bin_1 bin_2 ...
    bin_3 bin_4 bin1_idx bin2_idx bin3_idx bin4_idx n_bins_int sp test_size ...
    quant_range_basis 
if isempty(currData) 
    % turbine data
    currData = readmatrix('OWEZ240R75R.csv');
%     currData = readmatrix('OWEZ225R10N.xlsx');
    currData(1,:) = [];
%     currData(currData<0,:) = [];
%     currData(:,6:41) = [];
    % turbine diameter
    turbineDia = 90;
    % rated wind speed
    x1 = 15;
    % power production when wind speed is greater than rated wind speed
%     fPower = 3000;
    fPower = 1;
    % limit to below rated wind speed
    rWS = 15;
    % thrust Coefficient
    dat = readmatrix("Ct_V90.csv");
    dat_ws = dat(:,1);
    dat_Ct = dat(:,2);
    Ct=fit(dat_ws,dat_Ct,'smoothingspline');
    % test size
    test_size = 15;
    %% Sorting wind speed according to variance and creating bins
    theta0 = -100;
    % Bins are created according to qunatiles of wind speed
    % Size of each bin is different
%     sum_negative = sum(currData(:,6:end),2)>0;
%     currData = currData(sum_negative==1,:);
    sum_negative = sum(currData(:,6:end)<0,2);
    currData = currData(sum_negative == 0,:);
    currData(:,6:end) = currData(:,6:end)/max(max(currData(:,6:end)));
    [sort_ws, idx_ws] = sort(currData(:,1));
    sort_currData = currData(idx_ws,:);
    bin_1 = sort_currData((sort_currData(:,1)<=min(sort_ws)+(max(sort_ws) - min(sort_ws))/4),:);
    bin_2 = sort_currData((sort_currData(:,1)<=min(sort_ws)+2*(max(sort_ws) - min(sort_ws))/4) & (sort_currData(:,1)>min(sort_ws)+(max(sort_ws) - min(sort_ws))/4) ,:);
    bin_3 = sort_currData((sort_currData(:,1)<=min(sort_ws)+3*(max(sort_ws) - min(sort_ws))/4) & (sort_currData(:,1)>min(sort_ws)+2*(max(sort_ws) - min(sort_ws))/4) ,:);
    bin_4 = sort_currData((sort_currData(:,1)<=max(sort_ws)& (sort_currData(:,1)>min(sort_ws)+3*(max(sort_ws) - min(sort_ws))/4)),:);
    bin1_idx = idx_ws(1:length(bin_1));
    bin2_idx = idx_ws(length(bin_1)+1:length(bin_2)+length(bin_1));
    bin3_idx = idx_ws(length(bin_2)+length(bin_1)+1:length(bin_3)+length(bin_2)+length(bin_1));
    bin4_idx = idx_ws(length(bin_3)+length(bin_2)+length(bin_1)+1:length(bin_4)+length(bin_3)+length(bin_2)+length(bin_1));
    count_solver = 0;
    N = size(sort_currData,1);
    % Standard deviation of power generated by all the turbines within bins
    std_bins = [std2(bin_1(:,6:end)), std2(bin_2(:,6:end)), std2(bin_3(:,6:end)), std2(bin_4(:,6:end))];
    prob_bins = [size(bin_1,1)/N size(bin_2,1)/N size(bin_3,1)/N size(bin_4,1)/N];
    wt_bins = [std_bins(1)*prob_bins(1)/sum(prob_bins.*std_bins) std_bins(2)*prob_bins(2)/sum(prob_bins.*std_bins) std_bins(3)*prob_bins(3)/sum(prob_bins.*std_bins) std_bins(4)*prob_bins(4)/sum(prob_bins.*std_bins)];
    n_bins = wt_bins*test_size; % size of each bin (decimal)
    % Convert decimal size to integer and ensure that sum is equal to test size
    n_bins_int = floor(n_bins);
    if sum(n_bins_int) < test_size
        n_bins_int(1) = n_bins_int(1) + test_size - sum(n_bins_int);
    end
    %% Spline model for power curve
    % spline model based on all turbines in the first row
    % converting the data from a 1175x2x12 matrix to 14100x2 
    mdataPC = [];
    for i = 6:17
        mdataPC = [mdataPC; currData(:,1), currData(:,i)];
    end
    quant_range_basis = quantile(currData(:,1),[0.3333333333 0.6666666667]);
    % limit to below rated wind speed
    udataPC = mdataPC(mdataPC(:,1)<=rWS,:);
    % generate the basic spline model
    ws_basis_mat = WindSpeedBasisReg(udataPC(:,1), quant_range_basis);
    % regression using splines
    sp = fitlm(ws_basis_mat,udataPC(:,2));
end

if isempty(coordinate)
    % (x,y) coordinates of wind turbines
    coordinate = readmatrix('OWEZCoord.csv');
end

    bin1Stream = problemRng{1};
    bin2Stream = problemRng{2};
    bin3Stream = problemRng{3};
    bin4Stream = problemRng{4};

%% Calling the windcode
if size(runlength,2) == 1
if (runlength <= 0) || (runlength ~= round(runlength)) || (seed <= 0) || (round(seed) ~= seed)
    fprintf('runlength should be positive integer, seed must be a positive integer\n');
    fn = NaN;
    FnVar = NaN;
else 
    mean_err = zeros(runlength,1);
    for k = 1:runlength
    if size(theta,2) == 1
        if m_rep == 0
        bin1Stream.Substream = seed + k - 1;
        RandStream.setGlobalStream(bin1Stream);
        bin1_test_idx = bin1_idx(randperm(size(bin_1,1),n_bins_int(1)));
        bin1_test = currData(bin1_test_idx,1:2);

        bin2Stream.Substream = seed + k - 1;
        RandStream.setGlobalStream(bin2Stream);
        bin2_test_idx = bin2_idx(randperm(size(bin_2,1),n_bins_int(2)));
        bin2_test = currData(bin2_test_idx,1:2);

        bin3Stream.Substream = seed + k - 1;
        RandStream.setGlobalStream(bin3Stream);
        bin3_test_idx = bin3_idx(randperm(size(bin_3,1),n_bins_int(3)));
        bin3_test = currData(bin3_test_idx,1:2);

        bin4Stream.Substream = seed + k - 1;
        RandStream.setGlobalStream(bin4Stream);
        bin4_test_idx = bin4_idx(randperm(size(bin_4,1),n_bins_int(4)));
        bin4_test = currData(bin4_test_idx,1:2);

        % test data set
        test = [bin1_test; bin2_test; bin3_test; bin4_test]; 
        test_index = [bin1_test_idx; bin2_test_idx; bin3_test_idx; bin4_test_idx];
        test_actual = currData(test_index, 18:end); % only include turbines X13 - X36
        test_pred = windCode(test(:,1), test(:,2), theta, quant_range_basis, coordinate, turbineDia, Ct, sp, fPower, x1);
        else
        if m_rep >= 10
             load(strcat('testData0',int2str(m_rep),'.mat'));
        else
            load(strcat('testData00',int2str(m_rep),'.mat'));
        end
%         testData = readmatrix('WindDataSPlitTestwo');
%         testData(:,1) = [];
%         mData = mean2(testData(:,6:41));
%         stdData = std2(testData(:,6:41));
%         testData(:,6:41) = (testData(:,6:41) - mData)/stdData;
%         testData = currData; 
        binStream = problemRng{1};
        binStream.Substream = seed + k - 1;
        RandStream.setGlobalStream(binStream);
%         post_sample = datasample(testData,test_size);
        post_sample = testData;
        
        % test data set
        test = post_sample(:,1:2); 
        test_actual = post_sample(:,18:end);
        test_pred = windCode(test(:,1), test(:,2), theta, quant_range_basis, coordinate, turbineDia, Ct, sp, fPower, x1);
        test_size = size(post_sample,1);
        end
        err = zeros(1, test_size);
        % row 1 = 1:9
        % row 2 = 10:17
        % row 3 = 18:36
        for i = 1:test_size
            err(i) = mean((test_actual(i,18:end) - test_pred(i,18:end)).^2);
        end

        mean_err(k) = mean(err);    
        else
        % pick n_bins_int(i) data points from ith bin
    if cell2mat(theta(3)) == 0
        sample_data = cell2mat(theta(2));
        theta_o = cell2mat(theta(1));
        test = sample_data(:,1:2);
        test_actual = sample_data(:,18:end);
        test_pred = windCode(test(:,1), test(:,2), theta_o, quant_range_basis, coordinate, turbineDia, Ct, sp, fPower, x1);
        err = mean((test_actual-test_pred).^2);
        mean_err(k) = mean(err);
    else
        sample_data = cell2mat(theta(2));
        theta_o = cell2mat(theta(1));
        bin_no = cell2mat(theta(3));
        binStream = problemRng{bin_no};
        binStream.Substream = seed + k - 1;
        RandStream.setGlobalStream(binStream);
        sample_bin_data = datasample(sample_data, 1);
        r_ws = sample_bin_data(1);
        test = sample_bin_data(:,1:2);
        test_actual = sample_bin_data(:,18:end);
        test_pred = windCode(test(:,1), test(:,2), theta_o, quant_range_basis, coordinate, turbineDia, Ct, sp, fPower, x1);
        err = mean((test_actual-test_pred).^2);
        mean_err(k) = mean(err);
    end  
    end           
    end
    if runlength == 1
        fn = mean_err;
        FnVar = var(err)/runlength;
    else
        fn = mean(mean_err);
        FnVar = var(mean_err)/runlength;
    end 
end
else
    N = runlength;
    FnVar = NaN;
    bin1Stream.Substream = seed;
    RandStream.setGlobalStream(bin1Stream);
    bin1_test_idx = bin1_idx(randperm(size(bin_1,1),N(1)));
    bin1_test = currData(bin1_test_idx,1:2);

    bin2Stream.Substream = seed;
    RandStream.setGlobalStream(bin2Stream);
    bin2_test_idx = bin2_idx(randperm(size(bin_2,1),N(2)));
    bin2_test = currData(bin2_test_idx,1:2);

     bin3Stream.Substream = seed;
     RandStream.setGlobalStream(bin3Stream);
     bin3_test_idx = bin3_idx(randperm(size(bin_3,1),N(3)));
     bin3_test = currData(bin3_test_idx,1:2);

     bin4Stream.Substream = seed;
     RandStream.setGlobalStream(bin4Stream);
     bin4_test_idx = bin4_idx(randperm(size(bin_4,1),N(4)));
     bin4_test = currData(bin4_test_idx,1:2);

    test = [bin1_test; bin2_test; bin3_test; bin4_test]; 
    test_index = [bin1_test_idx; bin2_test_idx; bin3_test_idx; bin4_test_idx];
    test_actual = currData(test_index, 18:end); % only include turbines X13 - X36
    test_pred = windCode(test(:,1), test(:,2), theta(1), quant_range_basis, coordinate, turbineDia, Ct, sp, fPower, x1);
        err = zeros(1, sum(N(1:4)));
        for i = 1:sum(N(1:4))
            err(i) = (mean((test_actual(i,:) - test_pred(i,:)).^2));
        end
    fn = err;
end
end
%% Prediction
function yp = spPredict(xp, sp, fPower, x1, quant_range_basis)
% predict power using regression and splines 
% wind speeds are generated using wake model
yp = xp;
for u = 1:size(xp,2)
    w = WindSpeedBasisReg(xp(:,u), quant_range_basis);
    yp(:,u) = predict(sp,w);
end
yp(xp>=x1) = fPower;
end


%% Basis matrix for wind speeds
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
 
%%  wind wake model
function windSpeedHat = wake1(windSpeed, windAng, k, coordinate, turbineDia, Ct_fit)
% Set default incoming wind direction and rotate wind farm based on that
ang0 = 270;
Ct = Ct_fit(windSpeed);
Ct = min(Ct,1);
windAngR = (windAng - ang0)/180 *pi;
x1w = cos(windAngR)*(coordinate(:,1))' - sin(windAngR)*(coordinate(:,2))';
y1w = sin(windAngR)*(coordinate(:,1))' + cos(windAngR)*(coordinate(:,2))';
x1w = x1w - min(x1w,[],2);
y1w = y1w - min(y1w,[],2);
% Distances are measured in terms of turbine rotor diameter
x = x1w/turbineDia;
y = y1w/turbineDia;
% 2x the induction factor
aa = 1 - sqrt(1 - Ct);
% wake deficit variables
udef = zeros(size(x));
for i = 1:size(x,1) 
    currX = x(i,:);
    currY = y(i,:);
    % Initialize fractional overlap of rotor and wake areas (ignoring ground reflection)
    delX = currX' - currX;
    delY = abs(currY' - currY);
    
    % part of deficit formula
    def = aa(i)./(1 + 2*k*delX).^2;
    % remove turbines that are not in wake 
    def(delX <= 0) = 0;
    def(delY > (0.5 + k*delX)) = 0;
    
    % Geometry of wake overlap with rotor
    % distance between wake center and rotor center
    d = delY;
    rR = 0.5;
    wR = 0.5 + k*delX;
    
    % area overlap (circle - circle overlap)
    d2 = delY.^2;
    rR2 = rR.^2;
    wR2 = wR.^2;
    Ao = rR2.*acos((d2+rR2-wR2)./(2*d.*rR))+ wR2.*acos((d2-rR2+wR2)./(2*d.*wR))- 0.5*sqrt((-d+rR+wR).*(d-rR+wR).*(d+rR-wR).*(d+rR+wR));
    Ao((-d+rR+wR).*(d-rR+wR).*(d+rR-wR).*(d+rR+wR)<0) = 0;
    Ao(isnan(Ao)) = 0; % elements on diagonal will be NaN, since delY = 0 appears in denominator
    % fraction of rotor area overlapped by wake
    fo = Ao ./ (pi*rR2);
    fo(wR >= d + rR) = 1;
    udef(i,:) = (sum((fo.*def).^2,2)).^0.5;
end
windSpeedHat = windSpeed.*(1 - udef);
end

%% wind power model
function powerHat = windCode(xSeq, WDSeq, theta, quant_range_basis, coordinate, turbineDia, Ct, sp, fPower, x1)
windSpeedHat = wake1(xSeq, WDSeq, theta, coordinate, turbineDia, Ct);
powerHat = spPredict (windSpeedHat(:,13:36), sp, fPower, x1, quant_range_basis);
end