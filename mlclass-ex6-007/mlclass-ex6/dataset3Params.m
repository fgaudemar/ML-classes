function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


C_test_sample = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_test_sample = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
%pred_error = ones(size(C_test_sample), size(sigma_test_sample));
best_C = 0;
best_s = 0;
best_pred_error = 1000;
fprintf('finding best C and sigma parameters \n');


for c=1:size(C_test_sample)
  for s= 1:size(sigma_test_sample)
    
    C= C_test_sample(c);
    sigma = sigma_test_sample(s);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    pred_error = mean(double(predictions ~= yval));
    if pred_error < best_pred_error
      fprintf('better parameters found \n');
      best_pred_error = pred_error
      best_C = C
      best_s = sigma
    endif
  end
end

C = best_C;
sigma = best_s; 



% =========================================================================

end
