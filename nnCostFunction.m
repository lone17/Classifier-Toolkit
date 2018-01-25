function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

yMatrix = zeros(m, num_labels);
for i = 1:m
    yMatrix(i, y(i)) = 1;
end

X = [ones(m, 1) X];
a2 = sigmoid(X * Theta1');
a2 = [ones(m, 1) a2];
a3 = sigmoid(a2 * Theta2');
J = sum(sum(-yMatrix .* log(a3) - (1 - yMatrix) .* log(1 - a3))) / m;

J = J + lambda / 2 / m * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));

for i = 1:m
    delta3 = a3(i,:) - yMatrix(i, :);
    delta2 = delta3 * Theta2 .* a2(i,:) .* (1 - a2(i,:));
    Theta1_grad = Theta1_grad + delta2(:, 2:end)' * X(i,:);
    Theta2_grad = Theta2_grad + delta3' * a2(i, :);
end

Theta1_grad = (Theta1_grad + lambda * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)]) / m;
Theta2_grad = (Theta2_grad + lambda * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)]) / m;


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
