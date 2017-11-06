function demo_lasso()
	% clc
%     rng(10);
    addpath('utils/');
    addpath('spams/build');    
	d      = 10; 	% data dimension
	N      = 20; 	% number of samples 
	k      = 30; 	% dictionary size 
	lambda = 0.01;
	Y      = normc(rand(d, N));
	D      = normc(rand(d, k));
	%% cost function 
    function c = calc_F(X)
        c = (0.5*normF2(Y - D*X) + lambda*norm1(X))/size(X, 2);
    end
    %% fista solution 
	opts.pos = true;
	opts.lambda = lambda;
    opts.backtracking = false;
	X_fista = fista_lasso(Y, D, [], opts);
    %% fista with backtracking 
    opts.backtracking = true;
    opts.L0 = 1; 
    opts.eta = 1.5;
    X_fista_bt = fista_lasso_backtracking(Y, D, [], opts);
	%% spams solution 
	param.lambda     = lambda;
	param.lambda2    = 0;
	param.numThreads = 1;
	param.mode       = 2;
	param.pos        = opts.pos;
	X_spams      = mexLasso(Y, D, param); 

	%% compare costs 
	cost_spams = calc_F(X_spams);
	cost_fista = calc_F(X_fista);
    cost_fista_bt = calc_F(X_fista_bt);
    fprintf('Test lasso\n');
    fprintf('cost_spams                   = %.5s\n', cost_spams);
	fprintf('cost_fista                   = %.5s\n', cost_fista);
    fprintf('cost_fista with backtracking = %.5s\n', cost_fista_bt);
    
end