function demo_row_sparsity()
	clc
    addpath('utils');
    addpath('spams/build');
	d      = 30; 	% data dimension
	N      = 10; 	% number of samples 
	k      = 50; 	% dictionary size 
	lambda = 0.001;
	Y      = normc(rand(d, N));
	D      = normc(rand(d, k));
	%% cost function 
    function c = calc_F(X)
        c = 0.5*normF2(Y - D*X) + lambda*norm12(X);
    end
    %% fista solution 
	opts.pos = true;
	opts.lambda = lambda;
    opts.check_grad = 0;
    X_fista = fista_row_sparsity(Y, D, [], opts);
    cost_fista = calc_F(X_fista);
    fprintf('cost_fista    = %.5s\n', cost_fista);
    imagesc(X_fista)
end