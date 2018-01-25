clear; close all;

input_layer_size = 8;
hidden_layer_size = 30;
num_labels = 11;

% load data
Data = csvread('RD-1P.csv');

% number of train samples
m = 8000;
% number of test samples
m_test = 1000;

% Randomly pick the train set and test set
idx = randperm(size(Data, 1));

X = Data(idx(1:m), 1:end-1);
X_test = Data(idx(m+1 : m+m_test), 1:end-1);

y = Data(idx(1:m), end);
y_test = Data(idx(m+1 : m+m_test), end);

% X = [X , X.^2];
% X_test = [X_test , X_test.^2];

for i = 1:num_labels
    fprintf('Number of type %i: %i\n', i, sum(y == i));
end

% free up memory
clear Data

% normalize X
[X mu sigma] = featureNormalize(X);
X_test = (X_test - mu) ./ sigma;

% label start from 1
y += 1;
y_test += 1;

% initialize network parameter
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('\nTraining Neural Network... \n')

% max iteration
num_iter = 100;
options = optimset('MaxIter', num_iter);

% Weight regularization parameter
lambda = 0;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Accuracy on training set
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

% Accuracy on test set
pred_test = predict(Theta1, Theta2, X_test);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred_test == y_test)) * 100);

plot(1:num_iter, cost);
xlabel('Number of iterations');
ylabel('Cost');
