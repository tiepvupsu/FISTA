function res = norm12(X)
% function res = norm12(X)
% -----------------------------------------------
% Author: Tiep Vu, thv102@psu.edu, 6/10/2016 4:00:36 PM
%         (http://www.personal.psu.edu/thv102/)
% -----------------------------------------------
	res = sum(sqrt(sum(X.^2, 2)));
end 
