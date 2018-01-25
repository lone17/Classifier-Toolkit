function [J grad] = nnCostFunctionX(nn_params, layers_size, X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, layers_size, X, y, lambda)
%   computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Setup some useful variables
m = size(X, 1);
num_layers = numel(layers_size);

% Reshape nn_params back into the parameters
Theta = {};
for i = 1:num_layers-1
    Theta{i} = reshape(nn_params(1 : layers_size(i+1)*(layers_size(i)+1)),
                       layers_size(i+1), layers_size(i)+1);
    nn_params = nn_params(numel(Theta{i})+1 : end);
end

J = 0;
Theta_grad = {};
for i = 1:num_layers-1
    Theta_grad{i} = zeros(size(Theta{i}));
end

yMatrix = zeros(m, layers_size(end));
for i = 1:m
    yMatrix(i, y(i)) = 1;
end

a = {};
a{1} = X;
for i = 1:num_layers-1
    a{i} = [ones(m, 1) a{i}];
    a{i+1} = sigmoid(a{i} * Theta{i}');
end

J = sum(sum(-yMatrix .* log(a{end}) - (1 - yMatrix) .* log(1 - a{end}))) / m;
for i = 1:num_layers-1
    J += lambda / 2 / m * sum(sum(Theta{i}(:, 2:end) .^ 2));
end

delta = {};
for i = 1:m
    delta{num_layers} = a{num_layers}(i,:) - yMatrix(i, :);
    for j = num_layers-1 : -1 : 2
        if size(delta{j+1}) != size(Theta{j}, 1)
            delta{j} = delta{j+1} * [ones(1, size(Theta{j},2)); Theta{j}] .* a{j}(i,:) .* (1 - a{j}(i,:));
        else
            delta{j} = delta{j+1} * Theta{j} .* a{j}(i,:) .* (1 - a{j}(i,:));
        end
    end

    for j = 1:num_layers-2
        Theta_grad{j} = Theta_grad{j} + delta{j+1}(:, 2:end)' * a{j}(i,:);
    end
    Theta_grad{num_layers-1} = Theta_grad{num_layers-1} + delta{num_layers}' * a{num_layers-1}(i, :);
end

for i = 1:num_layers-1
    Theta_grad{i} = (Theta_grad{i} + lambda * [zeros(size(Theta{i}, 1), 1) Theta{i}(:, 2:end)]) / m;
end

% Unroll gradients
grad = [];
for i = 1:num_layers-1
    grad = [grad ; Theta_grad{i}(:)];
end


end
