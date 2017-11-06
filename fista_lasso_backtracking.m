function X = fista_lasso_backtracking(Y, D, Xinit, opts)

    if ~isfield(opts, 'backtracking')
        opts.backtracking = false;
    end 
    opts.regul = 'l1';
    opts = initOpts(opts);
    lambda = opts.lambda;

%     if numel(lambda) > 1 && size(lambda, 2)  == 1
%         lambda = repmat(opts.lambda, 1, size(Y, 2));
%     end
    if numel(Xinit) == 0
        Xinit = zeros(size(D,2), size(Y,2));
    end
    %% cost f
    function cost = calc_f(X)
        cost = 1/2 *normF2(Y(:, i) - D*X);
    end 
    %% cost function 
    function cost = calc_F(X)
        if numel(lambda) == 1 % scalar 
            cost = calc_f(X) + lambda*norm1(X);
         
        elseif numel(lambda) == numel(X)
            cost = calc_f(X) + norm1(lambda.*X);
        elseif numel(lambda) == size(X, 1) 
            lambda1 = repmat(lambda, 1, size(size(X, 2)));
            cost = calc_f(X) + norm1(lambda1.*X);
        end
    end 
    %% gradient
    DtD = D'*D;
    DtY = D'*Y;
    
    function res = grad(X) 
        res = DtD*X - DtY(:, i);
    end 
    % Checking gradient 
    if opts.check_grad
        check_grad(@calc_f, @grad, Xinit);
    end 

    opts.max_iter = 500;
    % for backtracking, we need to optimize one by one 
    X = zeros(size(Xinit));
    for i = 1:size(X, 2) 
       X(:, i) = fista_backtracking(@calc_f, @grad, Xinit(:, i), opts, ...
                                        @calc_F);
    end 
end 