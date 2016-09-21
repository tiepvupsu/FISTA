
# FISTA
**FISTA implementation in MATLAB based on the paper:**

A. Beck and M. Teboulle,  "A fast iterative shrinkage-thresholding algo-
rithm for linear inverse problems", *SIAM Journal on Imaging Sciences*,
vol. 2, no. 1, pp. 183–202, 2009. [View the paper](http://people.rennes.inria.fr/Cedric.Herzet/Cedric.Herzet/Sparse_Seminar/Entrees/2012/11/12_A_Fast_Iterative_Shrinkage-Thresholding_Algorithmfor_Linear_Inverse_Problems_(A._Beck,_M._Teboulle)_files/Breck_2009.pdf).

**Table of content**

<!-- MarkdownTOC -->

- [General Optimization problem](#general-optimization-problem)
- [Algorithms](#algorithms)
    - [If `L\(f\)` is easy to calculate,](#if-lf-is-easy-to-calculate)
    - [In case `L\(f\)` is hard to find,](#in-case-lf-is-hard-to-find)
- [Some typical `f\(x\)` functions](#some-typical-fx-functions)
- [Some typical `g\(x\)` functions](#some-typical-gx-functions)
    - [norm 1 \(LASSO\)](#norm-1-lasso)
    - [norm 2](#norm-2)
    - [row sparsity](#row-sparsity)
    - [group sparsity](#group-sparsity)

<!-- /MarkdownTOC -->


## General Optimization problem

`x = arg min F(x) = f(x) + lambda g(x)`                      (1)

where: 

- `g: R^n -> R`: a continuous convex function which is possibly _nonsmooth_. 
+ `f: R^n -> R`: a smooth convex function of the type `C^{1, 1}`, i.e., continously differentiable with Lipschitz continuous gradient `L(f)`:
`||grad_f(x) - grad_f(y)|| <= L(f)||x - y||` for every `x, y \in R^n`


## Algorithms

### If `L(f)` is easy to calculate,
We use the following algorithm:
![FISTA with constant step](https://raw.githubusercontent.com/tiepvupsu/FISTA/master/figs/FISTA_L.png)
where `pL(y)` is a proximal function as defined as:
![pL(y)](https://raw.githubusercontent.com/tiepvupsu/FISTA/master/figs/ply.png)

For a new problem, our job is to implement two functions: `grad_f(x)` and `pL(y)` which are often simpler than the original optimization stated in (1).

### In case `L(f)` is hard to find,
We can alternatively use the following algorithm: (in this version, I haven't implemented this):

![FISTA with backtracking](https://raw.githubusercontent.com/tiepvupsu/FISTA/master/figs/FISTA_noL.png)
where `QL(x, y)` is defined as:
![FISTA with backtracking](https://raw.githubusercontent.com/tiepvupsu/FISTA/master/figs/qlxy.png)

## Some typical `f(x)` functions

1. `f(x) = 0.5*||y - Dx||_F^2` 
If we let `DtD = D'*D, Dty = D'*y`, then we have:
    + `grad_f(x) = DtD*x - Dty`
    + `L(f) = max(eig(DtD))`

2. `f(x) = 0.5*x'*A*x + b'*x`
where `A` is a positive semidefinite, then: 
    + `grad_f(x) = A*x + b`
    + `L(f) = max(eig(A))`

## Some typical `g(x)` functions

### norm 1 (LASSO)
`g(x) = ||x||_1`
The corresponding `pL(x)` function would be like this:
`X = arg min_X 0.5||X - U||_F^2 + lambda||X||_1`
and can be broken down into: 
`x = arg min_x 0.5 (x - u)^2 + lambda x` (2)
Solution to this problem can be found at `proj/proj_l1.m`.

**Note**: the implemented function can work with weighted-LASSO problem and nonegative constraint as well. 
#### Usage 



### norm 2 
`g(x) = ||x||_2`

### row sparsity 
`X` is a matrix with many zero rows. 
`g(X) = sum ||X^i||_2` where `X^i` is the i-th row of `X`.
### group sparsity 