function X = proj_l12(U, opts)
% function X = proj_l12(U, opts)
% Description: Solve: xi = \arg\min 0.5*||xi - ui||_F^2 + lambda ||xi||_2
% where xi and ui are the i-th rows of X and U 
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
 	X = proj_l2(U', opts)';
    if nargin == 0 
        imagesc(X);
        X = [];
    end 
end
	