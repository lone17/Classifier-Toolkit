clear; close all;

layers_size = [7 15 15 15 11];
input_layer_size = layers_size(1);
num_labels = layers_size(end);
num_layers = numel(layers_size);

% load data
fprintf('\nLoading data...\n');
Data = csvread('RD-1P.csv');

% number of train samples
m = 8000;
% number of test samples
m_test = 1000;

fprintf('\nSetting up train samples and test samples...\n');
% Randomly pick the train set and test set
idx = randperm(size(Data, 1));

X = Data(idx(1:m), 2:end-1);
X_test = Data(idx(m+1 : m+m_test), 2:end-1);

y = Data(idx(1:m), end);
y_test = Data(idx(m+1 : m+m_test), end);

% X = [X , X.^2];
% X_test = [X_test , X_test.^2];

for i = 1:num_labels
    fprintf('Number of type %i: %i\n', i, sum(y == i));
end

% free up memory
clear Data

fprintf('\nPreprocessing samples...\n');
% normalize X
[X mu sigma] = featureNormalize(X);
X_test = (X_test - mu) ./ sigma;

% label start from 1
y += 1;
y_test += 1;

fprintf('\nInitializing network parameter...\n');
% initialize network parameter
initial_nn_params = [];
for i = 1:num_layers-1
    initial_nn_params = [initial_nn_params ;
                         randInitializeWeights(layers_size(i), layers_size(i+1))(:)];
end
% initial_Theta1 = randInitializeWeights(input_layer_size, layers_size(2));
% initial_Theta2 = randInitializeWeights(layers_size(2), num_labels);

% unroll parameters
% initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('\nTraining Neural Network... \n')

% max iteration
num_iter = 200;
options = optimset('MaxIter', num_iter);

% Weight regularization parameter
lambda = 0;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunctionX(p, layers_size, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta = {};
% Theta{1} = reshape(nn_params(1:layers_size(2) * (input_layer_size + 1)), ...
%                  layers_size(2), (input_layer_size + 1));
%
% Theta{2} = reshape(nn_params((1 + (layers_size(3) * (layers_size(2) + 1))):end), ...
%                  num_labels, (layers_size(2) + 1));
for i = 1:num_layers-1
    Theta{i} = reshape(nn_params(1 : layers_size(i+1)*(layers_size(i)+1)),
                       layers_size(i+1), layers_size(i)+1);
    nn_params = nn_params(numel(Theta{i})+1 : end);
end

% Accuracy on training set
pred = predictX(Theta, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

% Accuracy on test set
pred_test = predictX(Theta, X_test);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred_test == y_test)) * 100);

plot(1:num_iter, cost);
xlabel('Number of iterations');
ylabel('Cost');
