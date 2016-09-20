# FISTA
**FISTA implementation in MATLAB based on the paper:**

A. Beck and M. Teboulle,  "A fast iterative shrinkage-thresholding algo-
rithm for linear inverse problems", *SIAM Journal on Imaging Sciences*,
vol. 2, no. 1, pp. 183–202, 2009. [View the paper](http://people.rennes.inria.fr/Cedric.Herzet/Cedric.Herzet/Sparse_Seminar/Entrees/2012/11/12_A_Fast_Iterative_Shrinkage-Thresholding_Algorithmfor_Linear_Inverse_Problems_(A._Beck,_M._Teboulle)_files/Breck_2009.pdf).


## General Optimization problem    

`x = arg min F(x) = f(x) + lambda g(x)`

where: 

- `g: R^n -> R`: a continuous convex function which is possibly _nonsmooth_. 
+ `f: R^n -> R`: a smooth convex function of the type `C^{1, 1}`, i.e., continously differentiable with Lipschitz continuous gradient `L(f)`:
`||grad_f(x) - grad_f(y)|| <= L(f)||x - y||` for every `x, y \in R^n`


## Algorithms 

![FISTA with constant step](/figs/FISTA_L.png =200)

