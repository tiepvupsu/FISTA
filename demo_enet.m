function demo_enet()
	clc
	d      = 30; 	% data dimension
	N      = 70; 	% number of samples 
	k      = 50; 	% dictionary size 
	lambda = 0.01;
	lambda2 = 0.001;
	Y      = normc(rand(d, N));
	D      = normc(rand(d, k));
	%% cost function 
	function c = calc_F(X)
		c = 0.5*normF2(Y - D*X) + lambda2/2*normF2(X) + lambda*norm1(X);
	end
	%% fista solution 
	opts.pos = true;
	opts.lambda = lambda;
	opts.lambda2 = lambda2;
	opts.check_grad = 0;
	X_fista = fista_enet(Y, D, [], opts);
	%% spams solution 
	param.lambda     = lambda;
	param.lambda2    = lambda2;
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
