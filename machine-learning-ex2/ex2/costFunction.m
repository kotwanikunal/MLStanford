function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

summation_cost = 0;
for i = 1:m
	x = X(i,:);
	yi = y(i);
	summation_cost = summation_cost + ((-yi*log(sigmoid(theta'*x')))-((1-yi)*log(1-sigmoid(theta'*x'))));
endfor;

J = summation_cost/m;

for j=1:n
	summation_grad = 0;
	for i=1:m
		x = X(i,:);
		yi = y(i);
		summation_grad = summation_grad + ((sigmoid(theta'*x')-yi)*x(j));
	endfor;
	grad(j) = summation_grad/m;
endfor;


% =============================================================

end
