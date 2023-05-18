function [minmax, d, m, VarNature, VarBds, FnGradAvail, NumConstraintGradAvail, StartingSol, budget, ObjBd, OptimalSol, NumRngs] = WINDHOWLANDStructure(NumStartingSol)

minmax = -1; % minimize error (+1 for maximize)
d = 2; % Beta_i, i=0,1,2
m = 0; % no constraints
VarNature = zeros(d, 1); % real variables
VarBds = [0, 1; 0, 1]; % The betas are unconstrained
FnGradAvail = 0; % No Derivatives
NumConstraintGradAvail = 0; % No constraints
budget = 5000;
ObjBd = NaN; 
OptimalSol = NaN;
NumRngs = 100;
if (NumStartingSol == 0)
    StartingSol = NaN;
else
    StartingSol = [0.05 0.1];
%     StartingSol = rand(d, NumStartingSol)';
end