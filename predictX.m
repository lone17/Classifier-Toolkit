function p = predictX(Theta, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network

% Useful values
m = size(X, 1);
num_labels = size(Theta{end}, 1);
num_theta = numel(Theta);

p = zeros(size(X, 1), 1);

h = {};
h{1} = sigmoid([ones(m, 1) X] * Theta{1}');
for i = 2:num_theta
    h{i} = sigmoid([ones(m, 1) h{i-1}] * Theta{i}');
end

[dummy, p] = max(h{num_theta}, [], 2);

% =========================================================================


end
