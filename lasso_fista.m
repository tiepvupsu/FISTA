function [X, iter] = lasso_fista(Y, D, Xinit, opts)
% * Solving a Lasso problem using FISTA [[11]](#fn_fista): 
%           `X = arg min_X 0.5*||Y - DX||_F^2 + ||lambda*X||_1`. 
%   Note that `lambda` can be either a positive scalar or a matrix with 
%   positive elements.
% * Syntax: `[X, iter] = lasso_fista(Y, D, Xinit, lambda, opts)`
%   - INPUT:
%     + `Y, D, lambda`: as in the problem.
%     + `Xinit`: Initial guess 
%     + `opts`: options. 
%   - OUTPUT:
%     + `X`: solution.
%     + `iter`: number of fistat iterations.
% -----------------------------------------------
% Author: Tiep Vu, thv102@psu.edu, 4/6/2016
%         (http://www.personal.psu.edu/thv102/)
% -----------------------------------------------
    if nargin == 0 % test mode 
        clc;        
        addpath(fullfile('build'));
        addpath('proj');
        addpath('utils');
        d      = 300; % data dimension
        N      = 70; % number of samples 
        k      = 100; % dictionary size 
        lambda = 0.01;
        Y      = normc(rand(d, N));
        D      = normc(rand(d, k));
        Xinit  = zeros(size(D,2), size(Y, 2));
        %
        opts.max_iter     = 500;
        opts.verbose = false;
        opts.check_grad    = false;  
        opts.tol = 1e-8;    
        opts.lambda = lambda;
    end   
    %%
    lambda = opts.lambda;
    if numel(Xinit) == 0
        Xinit = zeros(size(D,2), size(Y,2));
    end
    %% 
    function cost = calc_f(X)
        cost = 1/2 *normF2(Y - D*X);
    end 
    %% cost function 
    function cost = calc_F(X)
        if numel(lambda) == 1 % scalar 
            cost = calc_f(X) + lambda*norm1(X);
        elseif numel(lambda) == numel(X)
            cost = calc_f(X) + norm1(lambda.*X);
        end
    end 
    %% gradient
    DtD = D'*D;
    DtY = D'*Y;
    function res = grad(X) 
        res = DtD*X - DtY;
    end 
    %% Checking gradient 
    if nargin == 0 && opts.check_grad
        check_grad(@calc_f, @grad, Xinit);
    end 
    %%
    L = max(eig(DtD));
    
    % [X, iter] = fista(@grad, Xinit, L, opts, @calc_F);
    [X, iter, ~] = fista_general(@grad, @proj_l1, Xinit, L, opts, @calc_F);
    %% For test only 
    if nargin == 0
        fprintf('A toy example:\n')
        fprintf('Data dimension                : %d\n', size(Y, 1));
        fprintf('No. of samples                : %d\n', size(Y, 2));
        fprintf('No. of atoms in the dictionary: %d\n', size(D, 2));
        fprintf('=====================================================\n')
        fprintf('Lasso Weighted FISTA solution vs SPAMS solution,\n');
        fprintf(' both of the following values should be close to 0.\n');
        % param for mex
        param.lambda     = lambda; 
        param.lambda2    = 0;
        param.numThreads = 1; 
        param.mode       = 2; 
        % mex solution and optimal value 
        param.pos = true;
        opts.pos = param.pos;        
        X_fista = lasso_fista(Y, D, [], opts);
        % [X, iter, ~] = fista(@grad, @proj_l1, Xinit, L, opts, @calc_F);
        X_spams      = mexLasso(Y, D, param); 
        fprintf('1. average(norm1(X_fista - X_spams)) = %5f\n', ...
            norm1(X_fista - X_spams)/numel(X));
        costmex = calc_F(X_spams);
        costfista = calc_F(X_fista);
        fprintf('2. costfista - cost_spams           = %5f\n', ...
            costfista - costmex);  
        if costfista < costmex
            fprintf('FISTA has better cost.\n');
        else 
            fprintf('SPAMS has better cost.\n');
        end 
        %% lasso_weighted test
        lambda = rand(size(X)); 
        opts.lambda = lambda;
        opts.pos = false;
        X_fista = lasso_fista(Y, D, [], opts);
        param.lambda     = 1; 
        param.lambda2    = 0;
        param.numThreads = 1; 
        param.mode       = 2; 
        param.pos = opts.pos;
        W = lambda;
        % mex solution and optimal value 
        X_spams      = mexLassoWeighted(Y, D, W, param);
        fprintf('=====================================================\n')
        fprintf('Lasso Weighted FISTA solution vs SPAMS solution,\n');
        fprintf(' both of the following values should be close to 0.\n');
        fprintf('1. average(norm1(X_fista - X_spams)) = %5f\n', ...
            norm1(X_fista - X_spams)/numel(X_fista));
        costmex = calc_F(X_fista);
        costfista = calc_F(X_spams);
        fprintf('2. costfista - cost_spams           = %5f\n', ...
            costfista - costmex);    
        if costfista < costmex
            fprintf('FISTA has better cost.\n');
        else 
            fprintf('SPAMS has better cost.\n');
        end 
        %% with positive constraint on X 
        fprintf('================Positive Constraint===================\n')
        fprintf('Lasso Weighted FISTA solution vs SPAMS solution,\n');
        fprintf(' both of the following values should be close to 0.\n');
        % opts for fista
        opts.lambda = 0.1;
        opts.pos = true;
        X_fista = lasso_fista(Y, D, [], opts);
        % param for mex
        param.lambda     = opts.lambda; 
        param.lambda2    = 0;
        param.numThreads = 1; 
        param.mode       = 2; 
        param.pos = opts.pos;
        % mex solution and optimal value   
        X_spams      = mexLasso(Y, D, param); 
        fprintf('1. average(norm1(X_fista - X_spams)) = %5f\n', ...
            norm1(X_fista - X_spams)/numel(X));
        costmex = calc_F(X_spams);
        costfista = calc_F(X_fista);
        fprintf('2. costfista - cost_spams           = %5f\n', ...
            costfista - costmex);   
        if costfista < costmex
            fprintf('FISTA has better cost.\n');
        else 
            fprintf('SPAMS has better cost.\n');
        end 
        %% lasso_weighted test
        fprintf('================Positive Constraint===================\n')
        % opts for fista         
        lambda = rand(size(X)); 
        opts.lambda = lambda;
        opts.pos = true;
        X_fista = lasso_fista(Y, D, [], opts);
        % param for mex 
        param.lambda     = 1; 
        param.lambda2    = 0;
        param.numThreads = 1; 
        param.mode       = 2; 
        param.pos = opts.pos;
        W = opts.lambda;
        % mex solution and optimal value 
        X_spams      = mexLassoWeighted(Y, D, W, param);
        fprintf('Lasso Weighted FISTA solution vs SPAMS solution,\n');
        fprintf(' both of the following values should be close to 0.\n');
        fprintf('1. average(norm1(X_fista - X_spams)) = %5f\n', ...
            norm1(X_fista - X_spams)/numel(X_fista));
        costmex = calc_F(X_fista);
        costfista = calc_F(X_spams);
        fprintf('2. costfista - cost_spams           = %5f\n', ...
            costfista - costmex); 
        if costfista < costmex
            fprintf('FISTA has better cost.\n');
        else 
            fprintf('SPAMS has better cost.\n');
        end 
        X = [];
    end 
    

end 