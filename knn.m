X = dlmread('input.txt');
k = 3;

%tic
%D = sqrt(sum(X.^2,2) - 2 * X*X.' + sum(X.^2,2).');
%toc

tic
[idx, dist] = knnsearch(X, X, 'K', k, 'Distance', 'euclidean');
toc

dlmwrite('matlab.txt', dist,'delimiter', ' ', 'precision', '%.06f');
dlmwrite('matlab.txt', idx, '-append', 'delimiter', ' ');