function X = proj_l1(U, opts)
% function X = proj_l1(U, opts)
% Description:
%   Soft Thresoding function. Solve one of the following problems:
%       U is a matrix of size d x k 
%       lambda is a positive scalar, a column vector of a matrix 
%       1. if lambda is a scalar:
%            X = argmin_X 0.5*||X - U|| + lambda||X||_1 
%       2. if lambda is a matrix with the same size as U 
%           X = argmin_X 0.5*||X - U|| + ||lambda .* X||_1 
%       3. if lambda is a column vector, suppose W is a matrix whose each 
%           column is lambda, number of columns is the same as number of 
%           columns of U:
%           X = argmin_X 0.5*||X - U|| + ||W.* X||_1 
%       if `opts.pos = true`, then this function solves each of above problems 
%           with positive constraints.
% Inputs: U: double dense matrix d x k 
%         opts: struct 
%             opts.pos: positive constraint (default = false)
% Outputs: X: a full matrix in d x k
% -----------------------------------------------
% Author: Tiep Vu, thv102@psu.edu, 6/9/2016 12:00:35 PM
%         (http://www.personal.psu.edu/thv102/)
% -----------------------------------------------
    % if ~isfield(opts, 'lambda')
    %     lambda = opts;      
    % else 
    %     lambda = lambda;
    % end 
    %%
    if ~isfield(opts, 'pos')
        opts.pos = false;
    end 
    %%
    lambda = opts.lambda;
    if numel(lambda) > 1 && size(lambda, 2) == 1 % column vector 
        lambda = repmat(lambda, 1, size(U, 2));
    end 
    %%
    if opts.pos 
        X = max(0, U - lambda);
    else 
        X = max(0, U - lambda) + min(0, U + lambda);
    end
end