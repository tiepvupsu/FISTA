function res = norm1(X)
	%% ================== File info ==========================
	% Author		: Tiep Vu (http://www.personal.psu.edu/thv102/)
	% Time created	: Tue Jan 26 22:32:25 2016
	% Last modified	: Tue Jan 26 22:32:27 2016
	% Description	:
	%	return norm 1 of the input matrix X 	
	%% ================== end File info ==========================
	res = full(sum(abs(X(:))));
end
