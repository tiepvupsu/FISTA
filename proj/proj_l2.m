function X = proj_l2(U, opts)
% function X = proj_l2(U, opts)
% Description: Solve: xi = \arg\min 0.5*||xi - ui||_F^2 + lambda ||xi||_2
% where xi and ui are the i-th columns of X and U 
% -----------------------------------------------
% Author: Tiep Vu, thv102@psu.edu, 6/8/2016 3:36:06 PM
%         (http://www.personal.psu.edu/thv102/)
% -----------------------------------------------
	if nargin == 0 
		d = 1000;
		n = 1000;
		U = normc(rand(n, d))';        
		lambda = 1;
        opts.lambda = lambda;
	end 
	%%
    lambda = opts.lambda;
	if ~isfield(opts, 'pos')
		opts.pos = false;
	end 
	%%
	if lambda == 0 
		X = U;
	else 
		% norm2_cols = sqrt(sum(U.^2, 1));
        tmp = norm2_cols(U);
		k = max(1 - lambda./tmp, 0);
		if opts.pos 
			X = repmat(k, size(U, 1), 1).*max(0, U);
		else 
			X = repmat(k, size(U, 1), 1).*U;
		end 
    end 
    if nargin == 0 
        imagesc(X);
        X = [];
    end 
end
	