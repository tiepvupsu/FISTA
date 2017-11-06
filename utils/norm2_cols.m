function res = norm2_cols(X)
% function res = norm2_cols(X)
% return norm2 of each column of X 
% ******************************************************************************
% * Date created    : 11/06/17
% * Author          : Tiep Vu 
% * Date modified   : 
% ******************************************************************************


    res = sqrt(sum(X.^2, 1));
end
