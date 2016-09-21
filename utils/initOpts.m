function opts = initOpts(opts)
    %% ================== File info ==========================
    % Author            : Tiep Vu (http://www.personal.psu.edu/thv102/)
    % Time created      : Wed Jan 27 23:57:31 2016
    % Last modified     : Wed Jan 27 23:57:32 2016
    % Description       : 
    %     INPUT:
    %
    %     OUTPUT: 
    %
    %% ================== end File info ==========================
    %% show, max_iter, checkgrad
    if ~isfield(opts, 'verbose')
        opts.verbose = 0;
    end 
    %%
    if ~isfield(opts, 'check_grad')
        opts.check_grad = 0;
    end 
    %%
    if ~isfield(opts, 'max_iter')
        opts.max_iter = 100;
    end 
    %%
    if ~isfield(opts, 'showD')
        opts.showD = false;
    end 
    %%
    if ~isfield(opts, 'showX')
        opts.showX = false;
    end 
    %%
    if ~isfield(opts, 'show_cost')
        opts.show_cost = 0;
    end 
    %%
    if ~isfield(opts, 'tol')
        opts.tol = 1e-8;
    end 
    %%
    
end