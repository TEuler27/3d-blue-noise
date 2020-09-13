# 3D Blue Noise Solver

The following code is linked to my degree thesis.

# Optimizer class

The optimizer object can be initialized like this:
```
Opt = Optimizer(100, 1, rho)
```
where 100 is the number of atoms of the discrete measure, rho is the density function and 1 indicates that the problem will be solved in [-1, 1]<sup>3</sup>.
rho should accept data of shape (3, k, m) and return data of the form (k, m). The optimizer class has mainly two functions: solve and saveNetwork.

## solve

The solve function solves the 3D blue noise problem. It has the following parameters:

- wCond: function to be used as a criterion for convergence on the w-gradient
- XCond: function to be used as a criterion for convergence on the X-gradient
- verbose (default False): if True solve prints the state of the problem at each iteration
- returnErrors (default False): if True solve returns 3 lists:
    - list of 1-norm of the gradient at each iteration
    - list of 2-norm of the gradient at each iteration
    - list of the values of R(X, w) at each iteration
- tuning (default True): if False solve does not do the initial tuning (just for research purpose)
- max_iter (default 500): max number of iteration

## saveNetwork
  
The saveNetwork function saves the diagram of the actual power diagram in an .obj file. It is also able to intersect the diagram with simple closed surfaces (es. cylinder, sphere). it accepts the following parameters:

- filename (default 'network.obj'): path of the file to be saved
- isOutside (default None): function that accept a point of shape (1,3) and returns a Boolean representing whether the point is inside or outside the surface. This function is required when the network needs to be intersected with a surface (Es for a cilynder ```isOutside(x)``` is ```x[0] ** 2 + x[1] ** 2 > r ** 2```)
- intersect (default None): function that accepts two points x,y of shape (1, 3) where isOutside(x) is False and isOutside(y) is True and returns the intersection of the edge [x,y] with the surface. This function is required when the network needs to be intersected with a surface
