import time
import numpy as np
import matplotlib.pyplot as plt

from base_functions import *
from plot_functions import *



# setup parameters
param = Parameters()

# read initial domain coefficients
c_in = read_from_file("./min/lambda3/lambda3B8.0_ab.txt" ,0) 

# setup inital domain
domain = GeneralizedDomain(c_in,param)
domain.rescale_domain()               # rescale and redistribute collocation points
domain.set_t_col(t_init=None, do_node_repel=domain.param.do_node_repel, node_repel_n_max_iter=domain.param.node_repel_n_max_iter)               # redefine t_col points for equi.arc.dist
domain.set_domain_variables()         # set collocation and source points

# optimize domain
gradient_descent_optimization(c_in, GeneralizedDomain, param)


exit()




