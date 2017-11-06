function X = fista_row_sparsity(Y, D, Xinit, opts)
    opts = initOpts(opts);
    lambda = opts.lambda;

    if numel(lambda) > 1 && size(lambda, 2)  == 1
        lambda = repmat(opts.lambda, 1, size(Y, 2));
    end
    if numel(Xinit) == 0
        Xinit = zeros(size(D,2), size(Y,2));
    end
    %% cost f
    function cost = calc_f(X)
        cost = 1/2 *normF2(Y - D*X);
    end 
    %% cost function 
    function cost = calc_F(X)
        cost = calc_f(X) + lambda*norm12(X);
    end 
    %% gradient
    DtD = D'*D;
    DtY = D'*Y;
    function res = grad(X) 
        res = DtD*X - DtY;
    end 
    %% Checking gradient 
    if opts.check_grad
        check_grad(@calc_f, @grad, Xinit);
    end 
    %% Lipschitz constant 
    L = max(eig(DtD));
    %% Use fista 
    opts.max_iter = 500;
    [X, ~, ~] = fista_general(@grad, @proj_l12, Xinit, L, opts, @calc_F);
    
    % we can also replace @proj_l12 by a function provided by SPAMS:
    % mexProximalFlat(U, opts)
    %%
    % opts.lambda = opts.lambda
    % opts.regul = 'l1l2'
    %[X, ~, ~] = fista_general(@grad, @proj_l12, Xinit, L, opts, @calc_F);    
    
end 