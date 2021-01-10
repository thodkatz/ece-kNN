X  = dlmread('logs/input.txt');
v0 = dlmread('logs/v0_log.txt');
v1 = dlmread('logs/v1_log.txt');
v2 = dlmread('logs/v2_log.txt');

k = 3;

%tic
%D = sqrt(sum(X.^2,2) - 2 * X*X.' + sum(X.^2,2).');
%toc

tic
[idx, dist] = knnsearch(X, X, 'K', k, 'Distance', 'euclidean');
toc

matlab = cat(1,dist,idx);

%dlmwrite('logs/matlab.txt', dist,'delimiter', ' ', 'precision', '%.06f');
%dlmwrite('logs/matlab.txt', idx, '-append', 'delimiter', ' ');

if v0 ~= v1 ~= v2
    fprintf("Versions 0,1,2 different results\n");
    fprintf("Make sure that there aren't duplicate distance values\n");
else 
    fprintf("Versions 0,1,2 agree with each other!\n");
end

if matlab ~= v0
    fprintf("Version 0 validation error\n");
elseif matlab ~= v1
    fprintf("Version 1 validation error\n");
elseif matlab ~= v2
    fprintf("Version 2 validation error\n");
else
    fprintf("Validation success!\n");
end
