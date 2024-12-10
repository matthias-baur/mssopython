
########################
###    MSSOPython    ###
########################

MSSOPython: Magnetic Schrödinger Shape Optimization code in Python
Author: MATTHIAS BAUR 
Contact: matthias.baur@mathematik.uni-stuttgart.de



#######################
###  Dependencies   ###
#######################

- NumPy, SciPy and matplotlib

- This code uses the PyGSL Python wrapper package for GSL functions, and in particular the confluent hypergeometric U function from it.
Link: https://github.com/pygsl/pygsl

Note that PyGSL needs to be built with special functions enabled (testing build) otherwise the code won't work!



#######################
###    Main code    ###
#######################

### base_functions.py ###

- contains all base functionality: 
  - Parameter class: set all parameters of the different routines here, also which eigenvalue to optimize and field strength B
  - Domain classes: Domain (star-shaped as in Antunes'2012), GeneralizedDomain (two Fourier series for x,y-coords of boundary curve, default), Rectangle
  - solving eigenvalue problem with METHOD OF FUNDAMENTAL SOLUTIONS (direct_problem(...))
  - gradient_descent based shape optimization (gradient_descent_optimization(...))
  - reading and writing domain coefficient/eigenvalue files

### plot_functions.py ### 

- various plot functions to plot domains, most important routines: 
    - plot_domain(...) plots the outline of a domain class object
    - plot_all_domains_from_file(...) plots an animation of all domains in sequence from all domains listed in an _ab.txt domain coefficient file 

    
    
#######################
### Example scripts ###
#######################

### script_optimize.py ### 

- an example script for gradient descent shape optimization. the default settings lead to optimization of lambda_3 at B=8.0 
- runtime depends on machine, but should be around 10 min per iteration and total runtime until convergence should be a few hours

### script_plot.py ### 

- an example script for plotting domains



######################
### Minimizer data ###
######################

### Folder "\min" ### 

- Folders "\min\lambdaN":
contains all domain coefficient files (ending _ab.txt) and corresponding eigenvalues (ending _val.txt) for eigenvalue index N=1 to 7 and various values of field strength B used for the plots of the paper

- Files "lambdaN_min_full_val.txt"
conjectured lambda_N^*(B) values used for plots in the paper (dashed black lines)

- Files "lambdaN_disjoint_unions_val.txt"
optimal eigenvalues attained by disjoint unions of previous minimizers (using values from previous "lambdaN_min_full_val.txt" files)

