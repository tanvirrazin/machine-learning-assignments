function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h_of_theta = sigmoid(X*theta);
matrix_for_J_part_one = -1 * ((y .* log(h_of_theta)) + ((ones(m, 1) - y) .* log(ones(m, 1) - h_of_theta)));

J_part_one = sum(matrix_for_J_part_one) / m;

J_part_two = (sum(theta .^ 2) - theta(1)^2) * lambda / (2*m);

J = J_part_one + J_part_two;

grad(1) = sum((h_of_theta - y) .* X(:,1)) / m;

for j = 2:size(X, 2),
    grad(j) = (sum((h_of_theta - y) .* X(:,j)) / m) + (theta(j)*lambda/m);
end;


% =============================================================

end
