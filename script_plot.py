import numpy as np
import scipy
import matplotlib.pyplot as plt

from base_functions import *
from plot_functions import *



##########################
### plot single domain ###
##########################

filename = "./min/lambda3/lambda3B8.0_ab.txt"
lineno = 0

# setup parameters
param = Parameters()

# read initial domain coefficients
c_in = read_from_file(filename, lineno)

# setup inital domain
domain = GeneralizedDomain(c_in,param)
domain.rescale_domain()               # rescale and redistribute collocation points
domain.set_t_col(t_init=None, do_node_repel=domain.param.do_node_repel, node_repel_n_max_iter=domain.param.node_repel_n_max_iter)   
domain.set_domain_variables()  

# plot single domain
plot_domain(domain, param, [0.0*np.pi],  None ) 



#################################################
### play animation of all domains from a file ###
#################################################

filename = "./min/lambda3/lambda3B8.0_ab.txt" 
plot_all_domains_from_file(filename, GeneralizedDomain, param)

exit()
