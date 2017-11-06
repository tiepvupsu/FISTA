function test_lasso()
	% clc
    addpath('../utils/')
	d      = 300; 	% data dimension
	N      = 70; 	% number of samples 
	k      = 100; 	% dictionary size 
	lambda = 0.01;
	Y      = normc(rand(d, N));
	D      = normc(rand(d, k));
	%% cost function 
    function c = calc_F(X)
        c = 0.5*normF2(Y - D*X) + lambda*norm1(X);
    end
    %% fista solution 
	opts.pos = true;
	opts.lambda = lambda;
	X_fista = fista_lasso(Y, D, [], opts);
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
	fprintf('cost_fista = %.5s\n', cost_fista);
	fprintf('cost_spams = %.5s\n', cost_spams);
end