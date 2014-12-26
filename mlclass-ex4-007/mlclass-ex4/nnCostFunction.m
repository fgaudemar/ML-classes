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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = [ones(m,1) sigmoid(z2)]; %a2 matrix m, hidden_layer_size+1
z3 = a2 * Theta2';
pred = sigmoid(z3); %pred matrix m,num-labels


%computation of y(i) for cost function
Yi = zeros(num_labels,m);

for i=1:m
  for c=1:num_labels
    Yi(c,i)= (y(i)==c) ;
  end
end


% 1) feedforward

for i=1:m
  %cost function
  J = J + 1 / m * (- log(pred(i,:)) * Yi(:,i) - log(1-pred(i,:))* (1-Yi(:,i)))  ;

      %grad = 1 / m * X' * (pred - y);
%     il faut separer theta1_grad et theta2_grad
% check comment adapter la formule dessous
      %temp = theta; 
      %temp(1) = 0;   % because we don't add anything for j = 0  
      %grad = grad + lambda / m * temp;
end

% regularization

J= J + lambda / (2 * m) * ( sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

% 2) backpropagation




Delta2 = zeros(size(Theta2));
Delta1 = zeros(size(Theta1));

for i=1:m

delta3 = pred'(:,i) - Yi(:,i); %delta3 vector num_labels,1
%fprintf('size delta3 %f \n', size(delta3));
gprime = ones(hidden_layer_size+1,1);
gprime(2:end) = sigmoidGradient(z2(i,:)');
%fprintf('size gprime %f \n', size(gprime));
delta2 = (Theta2' * delta3) .* gprime;
%fprintf('size delta2 %f \n', size(delta2));

Delta2 = Delta2 + delta3 *a2(i,:);
%fprintf('size Delta2 %f \n', size(Delta2));
Delta1 = Delta1 + delta2(2:end)*a1(i,:);
%fprintf('size Delta1 %f \n', size(Delta1));
end

Theta1_grad = 1/m * Delta1;
Theta2_grad = 1/m * Delta2;

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m * Theta2(:,2:end);


%[proba, p] = max(predictions,[],2);   utiliser plus tard ?










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
