function [ XR, indSorted, unsort ] = rotate( X, a, conditionTurb )
%Rotate the turbine locations and sort

% Shift
X(:,1) = X(:,1) - X(conditionTurb,1);
X(:,2) = X(:,2) - X(conditionTurb,2);

% Rotate
alpha = (a - 307.1);
alpha = alpha * pi / 180;
R = [cos(alpha), -sin(alpha); sin(alpha), cos(alpha)];
XstoreSmall = X; XR = X;
for t=1:length(XstoreSmall(:,1));
    XR(t,:) = R * XstoreSmall(t,:)';
end

% Sort
[~, indSorted] = sort(XR(:,1)); 
unsorted = 1:length(indSorted);
unsort(indSorted) = unsorted;  
XR = XR(indSorted,:);

end

