function res = norm_group(X, X_range, opts)
% function norm_group(X, X_range, opts)
% calculate cost (norm12) of X, given groups describe in X_range.
% Example: if X is a vector, then result is:
%  sum (norm2(X(X_range(i) + 1:X_range(i+1))))
% If X is a matrix 
% if opts.col_by_col = 1, then output is a row vector, each element is norm_group
% of one column of X 
% if opts.col_by_col = 0, then output is a scalar which is sum of all values
% when opts.col_by_col = 1.
% -----------------------------------------------
% Author: Tiep Vu, thv102@psu.edu, 6/16/2016 3:25:09 PM
%         (http://www.personal.psu.edu/thv102/)
% -----------------------------------------------
	if nargin == 0
		X_range = [0, 2, 5];
		X = [(1:5)', (6:10)'];
		opts.col_by_col = 1;
	end 
	%%
	if nargin < 3 || ~isfield(opts, 'col_by_col')
		opts.col_by_col = 1;
	end 
	%%
	n_groups = numel(X_range) - 1;
	G = zeros(n_groups, size(X, 2));
	V = size(X, 3);	
	for g = 1: n_groups
		Xg = get_block_row(X, g, X_range);
		if V > 1 % tensor 
			Xg1 = permute(Xg, [1, 3, 2]);
            d = size(Xg, 1) * V;
			Xg = reshape(Xg1, d, size(Xg, 2));
		end 
		G(g, :) = sqrt(sum(Xg.^2, 1));
	end 

	res = sum(G, 1);
	if ~opts.col_by_col
		res = sum(res);
	end 
	if nargin == 0 
		X = rand(5, 2, 3);
		res = norm_group(X, X_range, opts);
		disp(res);
	end 
end 
