%==========================================================================
% ASTRO-DF (Adaptive Sampling Trust Region Optimization - Derivative Free)
%==========================================================================
% DATE
%        Spring 2020
%
% AUTHOR
%        Pranav Jain, Yunsoo Ha, Sara Shashaani, Kurtis Konrad.
%
% REFERENCE		
%        Sara Shashaani, Fatemeh S. Hashemi and Raghu Pasupathy (2018)
%		 ASTRO-DF: A Class of Adaptive Sampling Trust Region Algorithms
%        for Derivative-Free Stochastic Optimization 28(4):3145-3176
%
%==========================================================================
%
% INPUT
%        problem
%              Problem function name
%        probstructHandle
%              Problem structure function name 
%        problemRng
%              Random number generators (streams) for problems
%        solverRng
%              Random number generator (stream) for solver
%        numBudget
%              number of budgets to record, >=3; the spacing between
%              adjacent budget points should be about the same
%
%
% OUTPUT
%        Ancalls
%              An array (size = 'NumSoln' X 1) of budget expended
%        Asoln
%              An array (size = 'NumSoln' X 'dim') of solutions
%              returned by solver
%        Afn
%              An array (size = 'NumSoln' X 1) of estimates of expected
%              objective function value
%        AFnVar
%              An array of variances corresponding to
%              the objective function at A
%              Equals NaN if solution is infeasible
%        AFnGrad
%              An array of gradient estimates at A; not reported
%        AFnGardCov
%              An array of gradient covariance matrices at A; not reported
%        Aconstraint
%              A vector of constraint function estimators; not applicable
%        AConstraintCov
%              An array of covariance matrices corresponding to the
%              constraint function at A; not applicable
%        AConstraintGrad
%              An array of constraint gradient estimators at A; not
%              applicable
%        AConstraintGradCov
%              An array of covariance matrices of constraint gradient
%              estimators at A; not applicable
%
%==========================================================================

%% ASTRO-DF
function [Ancalls, Asoln, Afn, AFnVar, AFnGrad, AFnGradCov, ...
    Aconstraint, AConstraintCov, AConstraintGrad, AConstraintGradCov] ...
    = SOWC2(probHandle, probstructHandle, problemRng, solverRng, m_rep)

    %% Unreported
    AFnGrad = NaN;
    AFnGradCov = NaN;
    Aconstraint = NaN;
    AConstraintCov = NaN;
    AConstraintGrad = NaN;
    AConstraintGradCov = NaN;
    
    % Separate the two solver random number streams
    solverInitialRng = solverRng{1}; % RNG for finding initial solutions
    solverInternalRng = solverRng{2};
    % Get details of the problem and random initial solution
    RandStream.setGlobalStream(solverInitialRng);
    
    [minmax, dim, ~, ~, VarBds, ~, ~, x0, budgetmax, ~, ~, ~] = probstructHandle(1);
    %get an initial solution
    budgetini=30;
    [xini, varini, ~, ~, ~, ~, ~, ~] = probHandle(x0, budgetini, problemRng, 1, 0);
    budgetmaxPT = budgetmax;

    % Set the sampling rule
    type = 3; % an integer value from 1 to 4
    
    % See the subfunction 'Adaptive_Sampling' for more details
%     currData = readmatrix('OWEZ240R75R.csv');
    currData = readmatrix('MDedpDatasubR2.csv');
    currData(1,:) = [];
%     currData = readmatrix('OWEZ240R75.csv');
    sum_negative = sum(currData(:,6:end)<0,2);
    currData = currData(sum_negative == 0,:);
    currData(:,6:end) = currData(:,6:end)/max(max(currData(:,6:end)));
    solverInternalRng.Substream = 1;
    RandStream.setGlobalStream(solverInternalRng);
    idx = randperm(size(currData,1));
    trainData = currData(idx(1:round(0.7*size(currData,1))),:);
    testData = currData(idx(round(0.7*size(currData,1))+1:end),:);
    save(sprintf('testData%03d.mat',m_rep),'testData');

%     mData = mean2(trainData(:,6:41));
%     stdData = std2(trainData(:,6:41));
%     trainData(:,6:41) = (trainData(:,6:41) - mData)/stdData;
%     trainData(trainData < 0) = 0;
    sz = size(trainData, 1);
    bestErr = inf;
    bestCutPoint = [];
    % tree = fitrtree([currData(:,1) currData(:,4)], mean(currData(:,6:end),2),'MaxNumSplits', 10, 'PredictorNames',{'WS' 'TI'});
    for j = 1:500
        rng(j)
        idrand = randperm(sz);
        c_trainData = trainData(idrand(1:round(0.8*sz)),:) ; 
        c_testData = trainData(idrand(round(0.8*sz)+1:end),:) ;
%         tree = fitrtree(c_trainData(:,1),c_trainData(:,2),'MaxNumSplits', 3);
        tree = fitrtree(c_trainData(:,4),mean(c_trainData(:,6:end),2),'MaxNumSplits', 3);
        %view(tree,'Mode','graph')
        cvtree = crossval(tree);
        err_cv = zeros(1,10);
        b1 = [];
        b2 = [];
        b3 = [];
        b4 = [];
        nj = zeros(1,4);
        sig2j = zeros(1,4);
        pj = zeros(1,4);
        for tc = 1:10
            bestCutPoint = cvtree.Trained{tc,1}.CutPoint;
            b1 = c_testData(c_testData(:,4) < bestCutPoint(2), :);
            b2 = c_testData(c_testData(:,4) < bestCutPoint(1) & c_testData(:,4) >= bestCutPoint(2), :);
            b3 = c_testData(c_testData(:,4) < bestCutPoint(3) & c_testData(:,4) >= bestCutPoint(1), :);
            b4 = c_testData(c_testData(:,4) >= bestCutPoint(3), :);
            nj = [size(b1,1) size(b2,1) size(b3,1) size(b4,1)];
            sig2j = [var(mean(b1(:,6:end),2)) var(mean(b2(:,6:end),2)) var(mean(b3(:,6:end),2)) var(mean(b4(:,6:end),2))];
            pj = nj/sum(nj);
            err_cv(tc) = sum((pj.*2).*sig2j./nj);
        end
        [err,err_id] = min(err_cv);
        if err < bestErr
            bestErr = err;
            BestCutPoint = cvtree.Trained{err_id,1}.CutPoint;
        end
    end
    b1 = trainData(trainData(:,4) < BestCutPoint(2),:);
    b2 = trainData(trainData(:,4) < BestCutPoint(1) & trainData(:,4) >= BestCutPoint(2),:);
    b3 = trainData(trainData(:,4) < BestCutPoint(3) & trainData(:,4) >= BestCutPoint(1),:);
    b4 = trainData(trainData(:,4) >= BestCutPoint(3),:);
    sm_strata = {b1; b2; b3; b4};
    nw = [size(b1,1) size(b2,1) size(b3,1) size(b4,1)];
    sig2j = [var(mean(b1(:,6:end),2)) var(mean(b2(:,6:end),2)) var(mean(b3(:,6:end),2)) var(mean(b4(:,6:end),2))];
    pw = nw/sum(nw);
    wi = pw.*sqrt(sig2j)/sum(pw.*sqrt(sig2j));

    
    Delta_max = norm(VarBds(:,2)-VarBds(:,1),inf); % maximum acceptable trust region radius
    if Delta_max==inf
        Delta_max=100;
    end

    Delta0 = .08*Delta_max; % initial trust region radius 
    shrink=.5^log(dim+1);
    expand=1/shrink;
    radiustry=[1 shrink expand]; %fraction of initial trust region to use for parameter tuning  
    Delta1 = [Delta0 Delta0*shrink Delta0*expand];
    
    if dim < 4 
        setupprct=.01; %the initial percentage of the budget to use for finding a good initial radius
    else
        setupprct=.005; %the initial percentage of the budget to use for finding a good initial radius
    end
    
    ptdict=[];
    ptdict_par = [];
    x_points=[];
    callcount=[];
    func_points =[];
    var_points = [];
    calls=0;
    funcbest=zeros(size(radiustry));
    xbest=zeros(length(radiustry),dim);    
    point_precision = 7; %number of decimal places to keep for any points in ptdict

    K = 4;
    ptdict = struct('pts',[x0],'counts',zeros(1,K),'means',[0],'variances',[0],...
        'rands',[1],'decimal',point_precision, 'iterationNumber', [0 0 0],...
        'means_bins',zeros(1,K),'variances_bins',zeros(1,K),'weights',[wi]);


    for i=1:length(radiustry)
        %try to run the algorithm on setupprct of the budget at different
        %fractions of the initial suggested radius
         [callcounti, x_pointsi, func_pointsi, var_pointsi, ptdict]= ASTRDF_Internal(...
            probHandle, problemRng, solverRng, minmax, dim, VarBds, x0, ...
            floor(setupprct*budgetmaxPT), type, Delta1(i), Delta_max, ptdict, 1, 0, K, trainData, pw, wi, sm_strata);        
        
        
        infoi=ptdict.info;
        ptdict=rmfield(ptdict,'info'); %only save the points
        x_pointsi_{i} = x_pointsi;
        func_pointsi_{i} = func_pointsi;
        callcounti_{i} = callcounti;
        var_pointsi_{i} = var_pointsi;
 
        calls=calls+infoi.calls; %total number of calls
                
        if ~isempty(func_pointsi) %if the attempt had a successful iteration
            %use the last point
            funcbest(i)=func_pointsi(end);
            xbest(i,:)=x_pointsi(end,:);            
            Delta_par(i) = ptdict.PTinfo(i).Delta;
        else
            if minmax==-1 %minimzation
                funcbest(i)=Inf; %no success means value was Inf
            elseif minmax==1
                funcbest(i)=0;
            end
            xbest(i,:)=x0;
            Delta_par(i) = Delta1(i);
        end
        
        st = ptdict;    
        st.xinc = xbest(i,:);
        save(sprintf('dictitertry3%06d%04d.mat',i),'st');
        
    end
    
    %pick the best value from the trials
    funcbest = -1*minmax*funcbest;
    if minmax == -1
        [bestval,best] = min(funcbest);
    elseif minmax == 1
        [bestval,best] = max(funcbest);
    end
    
    BestS = 0;
    
    for i = 1:3
        if best == i
            BestS = i;    
            x_aft_tune = xbest(i,:);
            Delta = Delta_par(i);
            x_points_par = x_pointsi_{i};
            func_points_par = cell2mat(func_pointsi_(i));
            callcount_par = cell2mat(callcounti_(i)) + budgetini + (2*floor(setupprct*budgetmaxPT));
            var_points_par = cell2mat(var_pointsi_(i));
            break
        end
    end
    
    sti = ptdict;
    sti.solverRng = solverRng;
    sti.BestS = BestS;
    sti.x_points_par = x_points_par;
    sti.Delta = Delta;
    %save(sprintf('dictitertry3%06d%04d.mat',10),'sti');

    %run the main algorithm
    [callcount, x_points, func_points, var_points]= ASTRDF_Internal(probHandle,...
        problemRng, solverRng, minmax, dim, VarBds, x_aft_tune, budgetmaxPT-3*floor(setupprct*budgetmaxPT), type, Delta, Delta_max, ptdict, 0, BestS, K, trainData, pw, wi, sm_strata);
    
    callcount = callcount+3*floor(setupprct*budgetmaxPT);
    
    %record points for new SIMOPT format Jan 2020
     Asoln = [x0; x_points_par; x_points];
     Afn = [xini; func_points_par; func_points];
     AFnVar = [varini; var_points_par; var_points];
     Ancalls = [budgetini; callcount_par; callcount];

end


function [callcount, x_points, func_points, var_points, ptdict] ...
    = ASTRDF_Internal(probHandle, problemRng, solverRng, ...
        minmax, dim, VarBds, xk, budgetmax, type, Delta0, Delta_max, ptdict, PT, BestS, K, trainData, pw, wi, sm_strata)
%ASTRDF_Internal runs the main portion of the ASTRO-DF Algorithm
%
%   INPUTS
%
%   probHandle = the problem handle
%   problemRng = Random number generators (streams) for problems
%   solverRng = Random number generator (stream) for solver
%   minmax = +/- 1 whether the goal is minimization or maximization
%          = 1 if problem is being maximized
%          =-1 if objective is being minimized
%   dim = the dimension of the problem
%   VarBds = dim x 2 array of the lower and upper bounds for each input
%               variable dimension
%   x0 = the initial point that the algorithm uses
%   budgetmax = the maximum number of function calls allowed
%   type = an integer used to determine the adaptive sample size rule
%               see Adaptive_Sampling function for details
%   Delta0 = the initial trust region radius size
%   Delta_max = the maximum allowed trust region radius
%   ptdict = (optional) the data dictionary structure that keeps track of visited points
%           .pts = the data points that have been visited
%           .counts = the number of times the function has been called at
%                           that point
%           .means = the mean values that the function has been called at a
%                           point
%           .variances = the variance values of the function at a point
%           .rands = the next random seed to use at a point
%           .decimal = the number of places after the decimal point to
%                           round values off
%           (optional inclusions for warmstarting)
%           .x_points = the visited incumbent solutions
%           .callcount = the number of calls at the incumbent solutions
%           .func_points = the mean function values at the incumbent solutions
%           .var_points = the variances at the incumbent solutions
%           .info = information structure with fields:
%                   .calls = the number of calls already made
%                   .iteration_number = the number of iterations already made
%   PT = the binary variable 
%        if PT = 0, main run
%        if PT = 1, run for parameter tuning
%   BestS = the discrete variable which shows the best scenario
%
%
%   OUTPUTS
%
%   callcount = an array of the number of calls needed to reach the
%                   incumbent solutions
%   x_points = an array of the incumbent solutions
%   func_points = the estimated function value at the incumbent solutions
%   var_points = the estimated variance at the incumbent solutions
%   ptdict = a data dictionary structure.  It contains the same information
%                   as before with these additional fields:
%           .info.delta = the current radius
%           .info.gradnorm = the norm of the current gradient
%           .sensitivity = the sensitivity of the variable bounds

    % Separate the two solver random number streams
    solverInitialRng = solverRng{1}; % RNG for finding initial solutions
    solverInternalRng = solverRng{2}; % RNG for the solver's internal randomness

    % Generate new starting point x0 (it must be a row vector)
    RandStream.setGlobalStream(solverInitialRng);
    % Numerical example 1
%     pw = [0.462 0.072 0.39 0.076];
%     wi = [0.4577 0.0395 0.4525 0.0503];

    % Numerical example 3
%     pw = [0.008 0.1140 0.6520 0.2260];
    % wi = [0.0066 0.1553 0.4989 0.3392];
    
                % Numerical example 4
%     pw = [0.0800 0.1190 0.2480 0.5530];
%     wi = [0.0192 0.0400 0.1661 0.7747];
    
    % More default values
    eta_1 = 0.10;           %threshhold for decent success
    eta_2 = 0.50;           %threshhold for good success
    w = 0.9; 
    mu = 100; 
    beta = 10;
    gamma_1 = (1.25)^(2/dim);  %successful step radius increase
    gamma_2 = 1/gamma_1;    %unsuccessful step radius decrease
  
    traj = [];
    budg_traj = [0];
    theta_traj = [0.25];
    Delta_traj = [0.8];
    sig2_traj = [0];
    gradnorm_traj = [0];
    N_traj = [0];
    
    %create the output variables or load them if available
    %if nargin< 12  || ~isfield(ptdict,'info')
    x_points=[];
    callcount=[];
    func_points =[];
    var_points = [];
    x_incum = [];
    call_incum = [];
    Fmean_incum = [];
    Fvar_incum = [];
    %Initializations
    calls = 0;
    
    if PT == 1
        iteration_number = 1;
    else
        iteration_number = ptdict.iterationNumber(BestS);
    end

    % Shrink VarBds to prevent floating errors
    %following STRONG code
    sensitivity = 10^(-5); % shrinking scale for VarBds
    VarBds(:,1) = VarBds(:,1) + sensitivity; 
    VarBds(:,2) = VarBds(:,2) - sensitivity;
    ptdict.sensitivity = sensitivity;
    
    Delta = Delta0;
    while calls <= budgetmax
        Delta1 = Delta;
%         o = 20;
%         if Delta > Delta_max/o       %if Delta > 1.2
%             lin_quad = 1; %run a linear model
%         else
%             lin_quad = 2; %run a quadratic model
%         end
        
        if PT == 1
            lin_quad = 1; %run a linear model
        else
            if iteration_number == ptdict.iterationNumber(BestS)
                lin_quad = 1;
            else
                lin_quad = 2; %run a quadratic model
            end
        end
        %run the adaptive sampling part of the algorithm
        [q, Fbar, Fvar, Deltak, calls, ptdict, budgetmax, Y, wi_xk] = Model_Construction(probHandle, xk, Delta, iteration_number, ...
            type, w, mu, beta, calls, solverInternalRng, problemRng, minmax, ptdict, VarBds,lin_quad,budgetmax, PT, BestS, dim, K, wi, pw, sm_strata, trainData);
        
        % Record Fbar
        x_incumbent = xk;
        Fbar_incumbent = Fbar(1);
        Fvar_incumbent = Fvar(1);
        [~,Loc] = ismember(round(xk,ptdict.decimal),ptdict.pts,'rows');
        Call_xk = ptdict.counts(Loc);

               % Step 3
        % Minimize the constrained model
        fun = @(x) Model_Approximation(x-Y(1,:), lin_quad, q);
        nonlcon = @(x)disk(x, xk, Deltak);
        [~,~,H,~,~] = fun(xk);
        
        hessint = @(x,lambda) hessinterior(x,lambda,H);
        options.HessianFcn = hessint;
        options.OptimalityTolerance = Fvar_incumbent*0.1;
        
        A = [];
        b = [];
        Aeq = [];
        beq = [];
        lb = max(xk-Deltak,VarBds(:,1)');
        ub = min(xk+Deltak,VarBds(:,2)');
        maxfuncevals_fmincon = 1000;
        C_Tol = 1e-12;
        S_Tol = 1e-20;
        
        options = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp', ...
                    'SpecifyObjectiveGradient', true, 'SpecifyConstraintGradient',true,...
                    'MaxFunctionEvaluations', maxfuncevals_fmincon, 'ConstraintTolerance', C_Tol, ...
                    'StepTolerance', S_Tol);
                
        [x_tilde, ~, exitflag, ~] = fmincon(fun, xk, A, b, Aeq, beq, lb, ub, nonlcon, options); 
        %[x_tilde, ~, ~, ~] = fmincon(fun, xk, A, b, Aeq, beq, lb, ub, [], options); 
        
        [~,LocResultxk] = ismember(round(xk,ptdict.decimal),ptdict.pts,'rows');
        Fb_xk = ptdict.means(LocResultxk);
        
        %Step 4
        %load in the current point's information if it exists already in the
        %point dictionary
        [Result,LocResult] = ismember(round(x_tilde,ptdict.decimal),ptdict.pts,'rows');
        if Result == true
            Fb = ptdict.means(LocResult);
            Fb_bins = ptdict.means_bins(LocResult,:);
            sig2 = ptdict.variances(LocResult);
            sig2_bins = ptdict.variances_bins(LocResult,:);
            samplecounts = ptdict.counts(LocResult,:);
            problemseed = ptdict.rands(LocResult); %set to global random seed for use in the function
            if PT == 1
                budgetmax = budgetmax - sum(samplecounts);
            end
        else
            Fb = 0;
            Fb_bins = zeros(1,K);
            sig2 = 0;
            sig2_bins = zeros(1,K);
            samplecounts = zeros(1,K); 
            %set to global random seed for use in the function
            % Using CRN: for each solution, start at substream 1 for problemRng
            problemseed = 1; % Reset seed to get CRN across solutions
        end
        Na_p = [];
        tau2 = 0;
        while 1
            Na = Adaptive_Sampling(iteration_number, sig2, Deltak, type, size(x_tilde,2), wi_xk, samplecounts, wi, Fb);
            Nt = sum(samplecounts);
            if sum(samplecounts >= Na) == 4
                break
            else
                if sum(samplecounts) > 40 && calls > budgetmax
                    break
                end
                Nt = Nt + 1;
            end
        for k = 1:K
            if samplecounts(k) >= Na(k)
                continue
            end
            samplecounts(k) = samplecounts(k) + 1;
            [xi, ~, ~, ~, ~, ~, ~, ~] = probHandle([x_tilde sm_strata(k) k], 1, problemRng, samplecounts(k), 1);
            xi = -minmax*xi; % Account for minimization/maximization 
            problemseed = problemseed + 1; % iid observations
            calls = calls + 1;
            F_old = Fb;
            F_old_bins = Fb_bins(k);
            Fb_bins(k) = (samplecounts(k)-1)/samplecounts(k)*Fb_bins(k) + 1/samplecounts(k)*xi; %update mean
%             Fb = (sum(samplecounts)-1)/sum(samplecounts)*Fb + 1/sum(samplecounts)*xi; %update mean
            Fb = sum(Fb_bins.*pw);
            sig2_bins(k) = (samplecounts(k)-2)/(samplecounts(k)-1)*sig2_bins(k) + samplecounts(k)*(Fb_bins(k)-F_old_bins)^2; 
%             sig2 = (sum(samplecounts)-2)/(sum(samplecounts)-1)*sig2 + sum(samplecounts)*(Fb-F_old)^2; %update variance
            temp_sig2 = 0;
            tau2 = 0;
            for tot_var = 1:K
                temp_sig2 = temp_sig2 + (sig2_bins(tot_var)*pw(tot_var)^2/samplecounts(tot_var));
                tau2 = tau2 + (sig2_bins(tot_var)*pw(tot_var)^2/wi(tot_var));
            end
            sig2 = temp_sig2;
            if samplecounts(k) == 1
                sig2_bins(k) = 0;
            end
            if sum(samplecounts) == 1
                sig2 = 0;
            end
            if sum(samplecounts) == Nt
                break
            end
        end
        end
        
        if calls > budgetmax 
            if PT == 1
                if Delta0 == .08*Delta_max
                    ptdict.iterationNumber(1) = iteration_number;
                    ptdict.PTinfo(1).Delta = Delta;
                elseif Delta0 < .08*Delta_max
                    ptdict.iterationNumber(2) = iteration_number;
                    ptdict.PTinfo(2).Delta = Delta;
                else
                    ptdict.iterationNumber(3) = iteration_number;
                    ptdict.PTinfo(3).Delta = Delta;
               end                          
            end
        end
        
%         wi = pw.*sqrt(sig2_bins)/sum(pw.*sqrt(sig2_bins));
        Fbar_tilde = Fb;
        Fvar_tilde = sig2;

        %save the information to the point dictionary
        if Result == false
            ptdict.pts = [ptdict.pts; round(x_tilde,ptdict.decimal)];
            ptdict.means = [ptdict.means; Fb];
            ptdict.means_bins = [ptdict.means_bins; Fb_bins];
            ptdict.counts = [ptdict.counts; samplecounts];
            ptdict.variances = [ptdict.variances; sig2];
            ptdict.variances_bins = [ptdict.variances_bins; sig2_bins];
            ptdict.rands = [ptdict.rands; problemseed]; 
            ptdict.weights = [ptdict.weights; wi];
        else
            ptdict.means(LocResult) = Fb;
            ptdict.means_bins(LocResult,:) = Fb_bins;
            ptdict.variances(LocResult) = sig2;
            ptdict.variances_bins(LocResult,:) = sig2_bins;
            ptdict.counts(LocResult,:) = samplecounts;
            ptdict.rands(LocResult) = problemseed;
            ptdict.weights(LocResult,:) = [wi];
        end
        
%         if Fbar_tilde > min(Fbar)
%             Fbar_tilde = min(Fbar);
%             Fvar_tilde = ptdict.variances(Fbar_tilde == ptdict.means,:);
%             x_tilde = ptdict.pts(Fbar_tilde == ptdict.means,:);
%         end 
        
        % Step 5 - Model Accuracy
%         rho = (Fbar(1) - Fbar_tilde)/ (Model_Approximation(xk-xk,lin_quad,q) - Model_Approximation(x_tilde-xk,lin_quad,q));
        rho = (Fb_xk - Fbar_tilde)/ (Model_Approximation(xk-Y(1,:),lin_quad,q) - Model_Approximation(x_tilde-Y(1,:),lin_quad,q));

%         if calls <= budgetmax*1 && PT == 0 && calls > budgetmax*0.9
%             calls
%             xk
%         end
%         if rho == 0
%             rho
%         end
        
        % Step 6 - Trust Region Update step
        if PT == 0
        if rho >= eta_2 && calls < budgetmax %really good accuracy
            xk = x_tilde;
            Delta = min(gamma_1*Deltak, Delta_max); %expand trust region
            x_points = [x_points;x_tilde];
            callcount = [callcount; calls];
            func_points =[func_points; Fbar_tilde];
            var_points = [var_points; Fvar_tilde];
        elseif rho >= eta_1 && calls < budgetmax %good accuracy
            xk = x_tilde;
            Delta = min(Deltak, Delta_max); %maintain same trust region size 
            x_points = [x_points; x_tilde];
            callcount = [callcount; calls];
            func_points = [func_points; Fbar_tilde];
            var_points = [var_points; Fvar_tilde];
        else %poor accuracy
            Delta = min(gamma_2*Deltak, Delta_max); %shrink trust region
        end 
        else
        if rho >= eta_2 %really good accuracy
            xk = x_tilde;
            Delta = min(gamma_1*Deltak, Delta_max); %expand trust region
            x_points = [x_points;x_tilde];
            callcount = [callcount; calls];
            func_points =[func_points; Fbar_tilde];
            var_points = [var_points; Fvar_tilde];
        elseif rho >= eta_1  %good accuracy
            xk = x_tilde;
            Delta = min(Deltak, Delta_max); %maintain same trust region size 
            x_points = [x_points; x_tilde];
            callcount = [callcount; calls];
            func_points = [func_points; Fbar_tilde];
            var_points = [var_points; Fvar_tilde];
        else %poor accuracy
            Delta = min(gamma_2*Deltak, Delta_max); %shrink trust region
        end 
        end
        
        
        [~,currentgrad] = Model_Approximation(xk-Y(1,:),lin_quad,q);
        iteration_number = iteration_number + 1;   
         
%         x_incum = [x_incum; x_incumbent];
%         call_incum = [call_incum; Call_xk];
%         Fmean_incum = [Fmean_incum; Fbar_incumbent];
%         Fvar_incum = [Fvar_incum; Fvar_incumbent];
        if PT == 0
            budg_traj(end + 1, 1) = calls;
            theta_traj(end + 1, 1) = xk;
            Delta_traj(end + 1, 1) = Delta;
            gradnorm_traj(end + 1, 1) = currentgrad;
            [~,Loc] = ismember(round(xk,ptdict.decimal),ptdict.pts,'rows');
            sig2_traj(end + 1, 1) = ptdict.variances(Loc);
            N_traj(end + 1, 1) = sum(ptdict.counts(Loc,:));
            traj = [budg_traj theta_traj Delta_traj sig2_traj gradnorm_traj N_traj];
            save('Trajectory_tracker_0.25_n_RAYT_non_normal.mat','traj');
        end   


        [~,Loc] = ismember(round(x_tilde,ptdict.decimal),ptdict.pts,'rows');
        Call_xt = ptdict.counts(Loc);
        
        x_incum = [x_incum; x_tilde];
        call_incum = [call_incum; Call_xt];
        Fmean_incum = [Fmean_incum; Fbar_tilde];
        Fvar_incum = [Fvar_incum; Fvar_tilde];
        
        stt = ptdict;
        stt.iteration_number = iteration_number;
        stt.PT = PT;
        stt.delta0 = Delta1;
        stt.delta = Delta;
        stt.calls = calls;
        stt.gradnorm = norm(currentgrad);
        stt.xinc = x_incumbent;
        stt.minc = Fbar_incumbent;
        stt.vinc = Fvar_incumbent;
        stt.trx = x_incum;
        stt.trc = call_incum;
        stt.trfm = Fmean_incum;
        stt.trfv = Fvar_incum;
        
%         save(sprintf('dictiterASTRDFDD8%04d%04d.mat',iteration_number),'stt');
    end

    %save final information before exiting
    if isempty(x_points) == false
        x_points = [x_points;x_points(end,:)];
        callcount = [callcount; calls];
        func_points = [func_points; func_points(end)];
        var_points = [var_points; var_points(end)];
    end
    info.iteration_number = iteration_number;
    info.delta = Delta;
    info.calls = calls;
    info.gradnorm = norm(currentgrad);
    ptdict.info = info;
end

function [q, Fbar, Fvar, Deltak, calls, ptdict, budgetmax, Y, wi_xk] = Model_Construction(probHandle, x, Delta, k, type, ...
    w, mu, beta, calls, solverInternalRng, problemRng, minmax, ptdict,VarBds,lin_quad, budgetmax, PT, BestS, dim, K, wi, pw, sm_strata, trainData)
%Model_Construction creates a model in the trust region
%
%   INPUTS
%   
%   probHandle = the problem handle
%   x = the center point that the algorithm builds the model around
%   Delta = the trust region radius size
%   k = the iteration number
%   type = an integer used to determine the adaptive sample size rule
%               see Adaptive_Sampling function for details
%   w = trust region contraction coefficient
%   mu = determine if trust region is sufficiently small
%   beta = trust region contraction coefficient (a different one)
%   calls = the number of calls already made
%   solverInternalRng = Random number generator (stream) for internal solver
%   problemRng = Random number generators (streams) for problems
%   minmax = +/- 1 whether the goal is minimization or maximization
%          = 1 if problem is being maximized
%          =-1 if objective is being minimized
%   ptdict = (optional) the data dictionary structure that keeps track of visited points
%           .pts = the data points that have been visited
%           .counts = the number of times the function has been called at
%                           that point
%           .means = the mean values that the function has been called at a
%                           point
%           .variances = the variance values of the function at a point
%           .rands = the next random seed to use at a point
%           .decimal = the number of places after the decimal point to
%                           round values off
%   VarBds = dim x 2 array of the lower and upper bounds for each input
%               variable dimension
%   lin_quad = is the current model linear or quadratic
%            = 1 if the model is linear
%            = 2 if the model is quadratic
%   budgetmax = the maximum number of function calls allowed
%   PT = the binary variable 
%        if PT = 0, main run
%        if PT = 1, run for parameter tuning
%
%   OUTPUTS
%
%   q = the coefficient vector for the model
%   Fbar = the matrix of function values
%   Fvar = the matrix of function variances
%   Deltak = the expanded or contracted radius size
%   calls = the number of calls already made
%   ptdict = a data dictionary structure.  It contains the same information as before.

    [~,LocResult] = ismember(round(x,ptdict.decimal),ptdict.pts,'rows');
    wi_xk = ptdict.weights(LocResult,:);
    
    j = 1;
    while 1
        Deltak = Delta*w^(j-1);
        
        %get the set of points to use
        Y = Interpolation_Points(x, Deltak, solverInternalRng, ptdict, k+j/2,VarBds,lin_quad);

        %build the model
        [~,~,~,A] = Model_Approximation(Y-Y(1,:),lin_quad);
                 
        if PT == 0 
            if k == ptdict.iterationNumber(BestS)
                Y = ptdict.PTinfo(BestS).pts;
                [~,~,~,A] = Model_Approximation(Y-Y(1,:),lin_quad);
            end
        else           
            if ptdict.iterationNumber(1) == 0
                ptdict.PTinfo(1).pts = Y;
            elseif ptdict.iterationNumber(2) == 0
                ptdict.PTinfo(2).pts = Y;
            elseif ptdict.iterationNumber(3) == 0
                ptdict.PTinfo(3).pts = Y;
            end
        end
                
        p = size(Y, 1);
        Fbar = zeros(p, 1);
        Fbar_bins = zeros(p, K);
        Fvar = zeros(p, 1);
        Fvar_bins = zeros(p, K); 
        ks_counts = zeros(p, K);
        stddev2 = zeros(p,1);
        wi_st = zeros(p, 4);
        stddev2_bins = zeros(p, K);
        randseeds = zeros(p,1);
        pts_exist = zeros(p,1);
        Fcui = zeros(p, 1);

        for i = 1:p %for each point in Y
            %if it already exists, recall the information
            [Result,LocResult] = ismember(round(Y(i,:),ptdict.decimal),ptdict.pts,'rows');
            if Result == true
                Fb = ptdict.means(LocResult);
                Fb_bins = ptdict.means_bins(LocResult,:);
                sig2 = ptdict.variances(LocResult);
                sig2_bins = ptdict.variances_bins(LocResult,:);
                wi = ptdict.weights(LocResult,:);
                ks = ptdict.counts(LocResult,:);
                problemseed = ptdict.rands(LocResult); %set to global random seed for use in the function
                pts_exist(i) = 1;
                if PT == 1
                    budgetmax = budgetmax - sum(ks);
                end   
            else
                Fb = 0;
                Fb_bins = zeros(1, K);
                sig2 = 0;
                sig2_bins = zeros(1, K);
                ks = zeros(1, K);
                % Reset seed to get CRN across solutions
                problemseed = 1;  %set to global random seed for use in the function
            end
            Na_p = [];
            tau2 = 0;
            while 1
                Na = Adaptive_Sampling(k, sig2, Deltak, type, size(Y,2), wi_xk, ks, wi, Fb);
                Nt = sum(ks);
                if sum(ks >= Na) == 4
                    break
                else
                    if sum(ks)>40 && calls>budgetmax
                        break
                    end
                    Nt = Nt + 1;
                end
            for l = 1:K
                if ks(l) >= Na(l)
                    continue
                else
                ks(l) = ks(l) + 1;
                [xi, ~, ~, ~, ~, ~, ~, ~] = probHandle([Y(i,:) sm_strata(l) l], 1, problemRng, ks(l), 1);
                xi = -minmax*xi; % Account for minimization/maximization
                problemseed = problemseed + 1; % iid observations  
                calls = calls + 1;
                F_old = Fb;
                F_old_bins = Fb_bins(l);
%                 Fb = (sum(ks) - 1)/sum(ks)*Fb + 1/sum(ks)*xi;
                Fb_bins(l) = (ks(l) - 1)/ks(l)*Fb_bins(l) + 1/ks(l)*xi; 
                Fb = sum(Fb_bins.*pw);
%                 sig2 = (sum(ks) - 2)/(sum(ks) - 1)*sig2 + sum(ks)*(Fb - F_old)^2;
                sig2_bins(l) = (ks(l) - 2)/(ks(l) - 1)*sig2_bins(l) + ks(l)*(Fb_bins(l) - F_old_bins)^2;
                temp_sig2 = 0;
                tau2 = 0;
                for tot_var = 1:K
                    temp_sig2 = temp_sig2 + (sig2_bins(tot_var)*pw(tot_var)^2/ks(tot_var));
                    tau2 = tau2 + (sig2_bins(tot_var)*pw(tot_var)^2/wi(tot_var));
                end
                sig2 = temp_sig2;
                if ks(l) == 1
                    sig2_bins(l) = 0;
                end
                if sum(ks) == 1
                    sig2 = 0;
                end
                end
                if sum(ks) == Nt
                    break
                end

            end
            end
            Fbar(i) = Fb;
            Fvar(i) = sig2;
            Fbar_bins(i,:) = Fb_bins;
            Fvar_bins(i,:) = sig2_bins;
            Fcui(i) = Fb + 1.96*sqrt(sig2/problemseed);

            % save the point information
            if pts_exist(i) == 0
                ks_counts(i,:) = ks;
                stddev2(i) = sig2;
                stddev2_bins(i,:) = sig2_bins; 
                randseeds(i) = problemseed;
                wi_st(i,:) = wi;
            else
                ptdict.means(LocResult) = Fb;
                ptdict.means_bins(LocResult,:) = Fb_bins;
                ptdict.variances(LocResult) = sig2;
                ptdict.variance_bins(LocResult,:) = sig2_bins;
                ptdict.counts(LocResult,:) = ks;
                ptdict.rands(LocResult) = problemseed;
                ptdict.weights(LocResult,:) = wi;
            end
        end

        %add the new points to the dictionary
        ptdict.pts = [ptdict.pts; round(Y(pts_exist==0,:),ptdict.decimal)];
        ptdict.means = [ptdict.means; Fbar(pts_exist==0)];
        ptdict.means_bins = [ptdict.means_bins; Fbar_bins(pts_exist==0,:)];
        ptdict.counts = [ptdict.counts; ks_counts(pts_exist==0,:)];
        ptdict.variances = [ptdict.variances; stddev2(pts_exist==0)];
        ptdict.variances_bins = [ptdict.variances_bins; stddev2_bins(pts_exist==0,:)];
        ptdict.rands = [ptdict.rands; randseeds(pts_exist==0)];
        ptdict.weights = [ptdict.weights; wi_st(pts_exist==0,:)];

        %get model parameters
        if PT == 1
            q = A \ Fbar;
        else
            q = A \ Fbar;
        end
                
        if any(isnan(q))
            warnmsg = ['NAN values occurred for building model parameters.\n',...
                       'Please confirm that the problem correctly works for only one replication at a time.\n',...
                       'Hint: investigate the runlength variable in the problem code as ASTRDF sets runlength=1.'];   
            warning(sprintf(warnmsg))
        end
        
        j = j + 1;

        [~, grad0] = Model_Approximation(Y(1,:)-x,lin_quad, q);
        %check size of adaptive sample
        if Deltak <= mu*norm(grad0)
            break
        elseif calls > budgetmax            
            break
        end
    end
    %contract radius
    Deltak = min(max(beta*norm(grad0), Deltak), Delta);
end

function [Y] = Interpolation_Points(center, radius, solverInternalRng, ptdict, iterations,VarBds,lin_quad)
%   Interpolation_Points selects the initial points to be used for the adaptive sampling
%
%   INPUTS
%
%   center = the center point for the model
%   radius = the maximum allowed radius
%   solverInternalRng = Random number generator (stream) for internal solver
%   ptdict = (optional) the data dictionary structure that keeps track of visited points
%           .pts = the data points that have been visited
%           .counts = the number of times the function has been called at
%                           that point
%           .means = the mean values that the function has been called at a
%                           point
%           .variances = the variance values of the function at a point
%           .rands = the next random seed to use at a point
%           .decimal = the number of places after the decimal point to
%                           round values off
%   iterations = the number of iterations that have occurred
%   VarBds = dim x 2 array of the lower and upper bounds for each input
%               variable dimension
%   lin_quad = is the current model linear or quadratic
%            = 1 if the model is linear
%            = 2 if the model is quadratic
%
%   OUTPUTS
%
%   Y = the set of points to use for adaptive sampling
%

    [~, dim] = size(center);
   
    if lin_quad == 1
        Y = center;
        e = @(k,n) [zeros(k-1,1);1;zeros(n-k,1)];
        
        for i = 1:dim
            if center(i) + radius > VarBds(i,2)
                Y = [Y; center-(rand(1)+9)/10*radius*e(i,dim)'];
            else
                Y = [Y; center+(rand(1)+9)/10*radius*e(i,dim)'];
            end
            
            if center(i) + radius >= VarBds(i,2) && center(i) + radius <= VarBds(i,1)
                Y(end) = [];
                new_delta1 = center(i) - VarBds(i,1);
                new_delta2 = VarBds(i,2) - center(i);
                if new_delta2 < new_delta1
                    Y = [Y; center-(rand(1)+9)/10*new_delta1*e(i,dim)'];
                else                    
                    Y = [Y; center+(rand(1)+9)/10*new_delta2*e(i,dim)'];
                end
            end
        end
        
    elseif lin_quad == 2
        Y = center;
        e = @(k,n) [zeros(k-1,1);1;zeros(n-k,1)];
        for i = 1:dim
            if center(i) + radius > VarBds(i,2) && center(i) - radius < VarBds(i,1)                
                new_center(i) = (VarBds(i,1) + VarBds(i,2))/2;
                new_radius(i) = (VarBds(i,2) - new_center(i));
                new_delta(i) = (VarBds(i,2) - new_center(i));
                
            elseif center(i) + radius > VarBds(i,2)
                new_delta(i) = VarBds(i,2) - center(i);
                new_radius(i) = (new_delta(i) + radius)/2;
                new_center(i) = VarBds(i,2) - new_radius(i);
                         
            elseif center(i) - radius < VarBds(i,1) % Shifts the center point so that our trust region is entirely within bounds
                new_delta(i) = center(i) - VarBds(i,1); % Distance between current center and bound
                new_radius(i) = (new_delta(i) + radius)/2; % 
                new_center(i) = VarBds(i,1) + new_radius(i);
                
            else
                new_delta(i) = inf;
                new_radius(i) = inf;
                new_center(i) = center(i);
            end
        end
        % new_center
        if min(new_delta) == inf
            for i = 1:dim
                Y = [Y; center+radius*e(i,dim)'];
                Y = [Y; center-radius*e(i,dim)'];
            end
        else
            Y = new_center;
            for j = 1:dim
                if new_radius(j) == inf
                    Y = [Y; new_center+(rand(1)+9)/10*radius*e(j,dim)'];
                    Y = [Y; new_center-(rand(1)+9)/10*radius*e(j,dim)'];
                else
                    Y = [Y; new_center+(rand(1)+9)/10*new_radius(j)*e(j,dim)'];
                    Y = [Y; new_center-(rand(1)+9)/10*new_radius(j)*e(j,dim)'];
                end
            end
            
            for z = 1:2*dim+1
                if z == 1
                    min_dis = norm(center-Y(z,:));
                    tempt_row = 1;
                end
                tempt = norm(center-Y(z,:));
                
                if tempt < min_dis 
                    tempt_row = z;
                end
            end
            Y(tempt_row,:) = center;
        end
    end
end

function numsamples = Adaptive_Sampling(k, sig2, Delta, type, dim, wi_xk, ks, wi, Fb)
%   Adaptive_Sampling determines the adaptive sample size
%
%   INPUTS
%
%   k = the number of times the function has been sampled
%   sig2 = the sample variance at the point
%   Delta = the trust region radius
%   type = a parameter value used for selecting different methods of
%               adaptive sampling
%
%   OUTPUTS
%
%   numsamples = the minimum required number of samples to meet the criteria
%
%
    
    alphak = 1;
    n0 = 1;
    lambdak = 80*(log10(k))^1.5; 
%     lambdak = 10^(log10(10+dim))*(log(k+0.001))^1.5;


    kappa = 1;
%     kappa = 10^(0.5);
%     N=floor(max([40,lambdak,(lambdak*sig2)/((kappa^2)*Fb^2)]));

    N=floor(max([40,lambdak]));
    
%     if N == floor((lambdak*sig2)/((kappa^2)*Fb^2))
%         N
%     end
    if isempty(wi)
        wi = [0.25 0.25 0.25 0.25];
    end

    if sig2 > (Fb^2)*(kappa^2)/lambdak
        N = max(N,sum(ks) + 1);
    end
    c_wi = cumsum(wi);
    
    if N > sum(ks) && sum(ks) >= 40

            u = rand;
            numsamples = ks;
            nrat = ks(1:4)/sum(ks);
            tr_vec = wi - nrat > 0;
            idx = 1:4;
            tr_idx = idx(tr_vec);
            wup = wi(tr_vec);
            wup = wup/sum(wup);
            c_wup = cumsum([0,wup]);
            [~,st_idx] = histc(u, c_wup);
            numsamples(tr_idx(st_idx)) = numsamples(tr_idx(st_idx)) + 1;

    else
        if sum(ks) < 8
            numsamples = [2 2 2 2];
        else
            numsamples = [2 2 2 2] + round(wi_xk*(N - 8));
            if sum(numsamples) < 40
                if 40 - sum(numsamples) == 1
                    numsamples = numsamples + [1 0 0 0];
                elseif 40 - sum(numsamples) == 2
                    numsamples = numsamples + [1 1 0 0];
                elseif 40 - sum(numsamples) == 3
                    numsamples = numsamples + [1 1 1 0];
                else
                    numsamples = numsamples + [1 1 1 1];
                end
            end
        end
    end



    
end
    
function [c, ceq,c_grad,ceq_grad] = disk(x, center, radius)
%   dispoints does constrained optimization within a disk
%
%   INPUTS
%
%   x = the point
%   center = the center point
%   radius = the trust region radius
%
%   OUTPUTS
%
%   these are the outputs required by the nonlincon function of fmincon
%   refer to MATLAB's help documentation for details
%
%
    dist = x - center;
    c = dot(dist,dist) - radius^2;
    c_grad=2*dist;
    if iscolumn(c_grad) == false
        c_grad = c_grad';
    end
    ceq = [];
    ceq_grad=[];
    
end

function hess = hessinterior(x,lambda,H) 
%   hessinterior is a function that creates a hessian that meets constraints
%       refer to MATLAB nonlinear constrained optimization documentation for help
%
%   INPUTS
%
%   x = the point
%   lambda = a hessian lambda boundary structure
%   H = standard hessian at a point
%
%   OUTPUTS
%
%   hess = the hessian function
%
%
    dim=size(x,1);
    myeye=2*eye(dim);
    lamb=lambda.ineqnonlin;
    hess=H+lamb(1)*myeye;
    for i=2:size(lamb,1)
        hess=hess-lamb(i)*myeye;
    end
    
end
  
function [y, grad,hess,Xvec,xstar] = Model_Approximation(x, lin_quad, qcoef, return_optimal,return_gradhess)
    %build the gradient and the function value of the quadratic model
    %Also build a Vandermonde 'matrix' Xvec
%
% INPUTS
% x = a m x n matrix with m points to evaluate and n total dimensions
%
% lin_quad = a numeric parameter that indicates if a linear or a quadratic
%       model is in use
%           Default is 2. aka quadratic
%
% qcoef = an optional parameter that is a vector of size p=(n+1)*(n+2)/2
%       for the quadratic case and p=n+1 for the linear case
%           Default is all zeros
%
% return_optimal = an optional parameter that indicates if the function
%       should return the optimal solution xstar
%           Default is false
%
% return_gradhess = an optional parameter that indicates if the function
%       should return the gradient and hessian
%           Default is true
%
% OUTPUTS
% y = a m x 1 column vector of function values at each point in x
%
% grad = a m x n matrix that represents the gradient at each of m points
%
% hess = a n x n matrix that represents the hessian of the function
%       since we only allow quadratic and linear functions, hess is the
%       same for all m points
%
% Xvec = the m x p Vandermonde matrix row for each point.
%
% xstar = the global optimal point of the quadratic function assuming it
%       is strictly convex or concave.  Otherwise, it returns a point where
%       the function gradient is 0.

    
    [count,dim] = size(x);
    if nargin<2
        lin_quad=2;
    end
        
    if lin_quad==2
        p = (dim + 1)*(dim + 2)/2;
    elseif lin_quad==1
        p = (dim + 1);
    else
        error("Variable 'lin_quad' must be either 1 or 2.")
    end
    
    if nargin<5
        return_gradhess=true;
    end
    if nargin<4
        return_optimal=false;
    end
    if nargin<3
        qcoef=zeros(p,1);
    elseif iscolumn(qcoef) == false
        qcoef = qcoef';
    end
    
    if return_optimal==true && lin_quad==1
        error("You cannot find an optimal solution to a linear model.")
    end
    
    if size(qcoef,1) ~= p
        error("The size of the coefficient vector is not correct for the model being built.")
    end

    % build vandermonde matrix
    Xvec = zeros(count, p);
    Xvec(:,1) = 1;
    Xvec(:,2:(2 + dim - 1)) = x; 

    if lin_quad==2
        k = 1 + dim + 1;
        for i = 1:dim
            for j = i:dim
                Xvec(:,k) = 0;
                if i == j
                    Xvec(:,k) = x(:,i).*x(:,j);
                    Xvec(:,k) = Xvec(:,k)/2;
                end
                k = k+1;
            end
        end
    end
    
    %get function values
    y = Xvec*qcoef;
    
    if nargout==1  %if you only need the y value
        return
    end
    if return_gradhess==false && return_optimal==false
        grad=[];
        hess=[];
        return
    end
    
    beta=qcoef(2:dim+1);
    if lin_quad==2
        gamma=qcoef(dim+2:end);
        A=tril(ones(dim));
        A(A==1)=gamma;
        A=(A'+A)/2;
        A=A-diag(diag(A))/2; %correct the diagonals because the 1/2
        hess=A+A';
    elseif lin_quad==1
        hess=zeros(dim);
    end
    grad=beta+(hess)*x';
    grad=grad';
    
    if return_optimal==true
        if rcond(hess)>.001
            xstar=hess\(-1*beta); %the unconstrained optimal (set grad==0)
            xstar=xstar';
        else
            xstar=[];
        end
    else
        xstar=[];
    end
    
end