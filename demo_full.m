function X = demo_full()    
    addpath('spams/build');
    addpath('proj');
    addpath('utils');
    d      = 300; % data dimension
    N      = 70; % number of samples 
    k      = 100; % dictionary size 
    lambda = 0.01;
    Y      = normc(rand(d, N));
    D      = normc(rand(d, k));
    opts.max_iter   = 500;
    opts.verbose    = false;
    opts.check_grad = false;
    opts.tol        = 1e-8;
    opts.lambda     = lambda;
    %% cost function 
    function cost = calc_F(X)
        if numel(lambda) == 1 % scalar 
            cost = 1/2 *normF2(Y - D*X) + lambda*norm1(X);
        elseif numel(lambda) == numel(X)
            cost = 1/2 *normF2(Y - D*X) + norm1(lambda.*X);
        end
    end 
    fprintf('********************Full demo**********************\n');
    fprintf('A toy example:\n')
    fprintf('Data dimension                : %d\n', size(Y, 1));
    fprintf('No. of samples                : %d\n', size(Y, 2));
    fprintf('No. of atoms in the dictionary: %d\n', size(D, 2));
    fprintf('=====================================================\n')
    fprintf('Lasso FISTA solution vs SPAMS solution,\n');
    fprintf(' both of the following values should be close to 0.\n');
    % param for mex
    param.lambda     = lambda; 
    param.lambda2    = 0;
    param.numThreads = 1; 
    param.mode       = 2; 
    % mex solution and optimal value 
    param.pos = true;
    opts.pos = param.pos;        
    X_fista = fista_lasso(Y, D, [], opts);
    X_spams      = mexLasso(Y, D, param); 
    fprintf('1. average(norm1(X_fista - X_spams)) = %5f\n', ...
        norm1(X_fista - X_spams)/numel(X_spams));
    costmex = calc_F(X_spams);
    costfista = calc_F(X_fista);
    fprintf('2. costfista - cost_spams            = %5f\n', ...
        costfista - costmex);  
    if costfista < costmex
        fprintf('FISTA provides a better cost.\n');
    else 
        fprintf('SPAMS provides a better cost.\n');
    end 
    %% lasso_weighted test
    lambda = rand(size(X_spams)); 
    opts.lambda = lambda;
    opts.pos = false;
    X_fista = fista_lasso(Y, D, [], opts);
    
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
    fprintf('2. costfista - cost_spams            = %5f\n', ...
        costfista - costmex);    
    if costfista < costmex
        fprintf('FISTA provides a better cost.\n');
    else 
        fprintf('SPAMS provides a better cost.\n');
    end 
    %% with positive constraint on X 
    fprintf('================Positive Constraint===================\n')
    fprintf('Lasso FISTA solution vs SPAMS solution,\n');
    fprintf(' both of the following values should be close to 0.\n');
    % opts for fista
    opts.lambda = 0.1;
    opts.pos = true;
    X_fista = fista_lasso(Y, D, [], opts);
    % param for mex
    param.lambda     = opts.lambda; 
    param.lambda2    = 0;
    param.numThreads = 1; 
    param.mode       = 2; 
    param.pos = opts.pos;
    % mex solution and optimal value   
    X_spams      = mexLasso(Y, D, param); 
    fprintf('1. average(norm1(X_fista - X_spams)) = %5f\n', ...
        norm1(X_fista - X_spams)/numel(X_spams));
    costmex = calc_F(X_spams);
    costfista = calc_F(X_fista);
    fprintf('2. costfista - cost_spams            = %5f\n', ...
        costfista - costmex);   
    if costfista < costmex
        fprintf('FISTA provides a better cost.\n');
    else 
        fprintf('SPAMS provides a better cost.\n');
    end 
    %% lasso_weighted test
    fprintf('================Positive Constraint===================\n')
    % opts for fista         
    lambda = rand(size(X_spams)); 
    opts.lambda = lambda;
    opts.pos = true;
%         X_fista = lasso_fista(Y, D, [], opts);
    X_fista = fista_lasso(Y, D, [], opts);
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
    fprintf('2. costfista - cost_spams            = %5f\n', ...
        costfista - costmex); 
    if costfista < costmex
        fprintf('FISTA provides a better cost.\n');
    else 
        fprintf('SPAMS provides a better cost.\n');
    end 
    X = [];
end 
    

