function res = normF2(A)
    % compatible with multi-channel
	%% ================== File info ==========================
	% Author		: Tiep Vu (http://www.personal.psu.edu/thv102/)
	% Time created	: Tue Jan 26 22:33:47 2016
	% Last modified	: Tue Jan 26 22:33:48 2016
	% Description	: square of the Frobenius Norm of A 
	%% ================== end File info ==========================
    B = A.^2;
    B = B(:);
    res = sum(B);
	% res = norm(A, 'fro')^2;
end 