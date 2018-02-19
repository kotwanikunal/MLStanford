function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
no_of_rows = size(z,1);
no_of_columns = size(z,2);

for i = 1:no_of_rows
	for j= 1:no_of_columns
		g(i,j) = 1 / (1 + exp(-1 * z(i,j))); 
	endfor;
endfor;


% =============================================================

end
