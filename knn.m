X = dlmread('input.txt');
k = 3;

%tic
%D = sqrt(sum(X.^2,2) - 2 * X*X.' + sum(X.^2,2).');
%toc

tic
[idx, dist] = knnsearch(X, X, 'K', k, 'Distance', 'euclidean');
toc

file = fopen('matlab.txt', 'w');
fprintf(file, '[');
for k = 1:size(dist(:,1))
    fprintf(file, '[%f,%f,%f,%f,%f],\n', dist(k,:));
end
fprintf(file, ']\n');

fprintf(file, '[');
for k = 1:size(idx(:,1))
    fprintf(file, '[%d,%d,%d,%d,%d],\n', idx(k,:));
end
fprintf(file, ']\n');

fclose(file);