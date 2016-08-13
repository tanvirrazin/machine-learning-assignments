function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

%for indeex = 1:size(X,1)
%    prev_distan = 1000000;
%    distan = 0;
%    for ceni = 1:K
%        distan = sqrt(((X(indeex, 1)-centroids(ceni, 1))^2) + ((X(indeex, 2)-centroids(ceni, 2))^2));
%
%        if distan < prev_distan
%            idx(indeex) = ceni;
%            prev_distan = distan;
%        end;
%    end;
%end;

%for indeex = 1:size(X, 1)
%    distance_set = zeros(K, 1);
%    for ceni = 1:K
%        distance_set(ceni, 1) = sqrt(((X(indeex, 1)-centroids(ceni, 1))^2) + ((X(indeex, 2)-centroids(ceni, 2))^2));
%    end;

%    idx(indeex) = find(distance_set==min(distance_set));
%end;

for indeex = 1:size(X, 1)
    distance_matrix = repmat(X(indeex, :), K, 1);
    [minVal, minIndx] = min(sqrt(sum(((distance_matrix - centroids) .^ 2), 2)));
    idx(indeex) = minIndx;
end;


% =============================================================

end
