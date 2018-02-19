function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(X,2);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

summation_cost = 0;
for i = 1:m
	x = X(i,:);
	yi = y(i);
	summation_cost = summation_cost + ((-yi*log(sigmoid(theta'*x')))-((1-yi)*log(1-sigmoid(theta'*x'))));
endfor;
summation_cost = summation_cost/m;

summation_reg = 0;
for j = 2:n
	theta_j  = theta(j);
	summation_reg = summation_reg + (theta_j^2);
endfor;

summation_reg = ((summation_reg*lambda)/(2*m));

J = summation_cost + summation_reg;


summation_grad0 = 0;
	for i=1:m
		x = X(i,:);
		yi = y(i);
		summation_grad0 = summation_grad0 + ((sigmoid(theta'*x')-yi)*x(1));
	endfor;
	grad(1) = summation_grad0/m;

for j=2:n
	summation_grad = 0;
	for i=1:m
		x = X(i,:);
		yi = y(i);
		summation_grad = summation_grad + ((sigmoid(theta'*x')-yi)*x(j));
	endfor;
	grad(j) = (summation_grad/m)+((lambda/m)*theta(j));
endfor;


% =============================================================

end
