function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    h_of_theta_mat = X * theta;
    diff_mat = h_of_theta_mat - y;

    temp_theta_1 = theta(1) - (sum(diff_mat .* X(:, 1))*alpha/m);
    temp_theta_2 = theta(2) - (sum(diff_mat .* X(:, 2))*alpha/m);

    theta(1) = temp_theta_1;
    theta(2) = temp_theta_2;



    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);
    disp(J_history(iter));

end

end
