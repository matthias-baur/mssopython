import time
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D
import scipy.special
from scipy.special import hyperu
from scipy.special import hankel1
import mpmath 
from scipy.signal import argrelmax
from scipy.sparse.linalg import eigsh
from numpy.linalg import norm
from numpy.random import *
from pygsl.testing.sf import hyperg_U

debug = 0

###########################################################################
###          Base functions for optimization
###########################################################################

# Define Parameters class
# stores all global parameters
class Parameters:
    def __init__(self): #, n_, sum_length_, interval_end_, len_ab_, N_col_, dt_col_, p_beta_, eps_, n_steps_, max_steps_, beta_opt_, beta_rel_, mult_tol_):
        self.operator = "MDL"           # choose operator for optimization: DL     -    Dirichlet Laplacian 
                                       #                                   MDL    -    Magnetic Dirichlet Laplacian

        # Define sum that is used as objective function J:
        # J(lambda_1,...,lambda_n) = lambda_{n-sum_length+1} + ... + lambda_n
        self.n = 3;                    # index of largest eigenvalue involved
        self.sum_length = 1 #self.n;            # length of sum
        self.sum_weight = self.n*[1.0]        # weight of eigenvalues, lowest to highest, no of weights = sum_length
        self.p0 = 1                    # l^p0 norm which is applied to sum

        self.A0 = 1.0                  # area to which domains are rescaled
        self.B0 = 8.0                  # magnetic field strength    
        
        self.interval_start = max(18.037, self.B0)-0.0063
        self.interval_end = 60.8202          # upper limit for eigenvalues (choose slightly 
                                                # larger than lambda_n of initial domain for gradient descent)
                                                # needs to be larger than 4*pi*n
        self.Nk = 30*self.n                   # no. of logdetA evals
        self.n_Iter_fine_min = 0               # min number of finer mesh sing1A turns, 0 for gradient descent, 1 for genetic algorithm

        self.len_ab = 20;               # number of sine and cosine cofficients

        self.n_V_current = self.sum_length          # save how many eigenvalues were used for gradient in last iteration
        
        # Set basic parameters for collocation points 
        self.N_col  = 300   
        self.dt_col = np.array(self.N_col*[2*np.pi/self.N_col]);      # step width of angular paramter
        self.t_col = np.linspace(0,(2*np.pi-self.dt_col[0]),self.N_col)[np.newaxis]; 
        
        self.N_interior = 100
        self.N_p_source = self.N_col
        self.p_beta = 0.017;             # displacement distance for MFS eigenfunction source points
                                        # along normals, corresponds to \delta_y in the report
                                        # recommendations: 
                                        # genetic algorithm     p_beta = 0.01;
                                        # gradient descent      p_beta = 0.03;
                                        
        # node repel parameters
        self.node_repel_p = 3                # exponent of pair potential
        self.node_repel_n_max_iter = 60     # max number of iterations in BGFS
        self.do_node_repel = False #True          # do node repel algorithm?
                                        
       
        # integration fineness
        self.N_int  = 2**11
        self.dt_int  = 2*np.pi/self.N_int;
        self.t_int = np.linspace(0,(2*np.pi-self.dt_int),self.N_int)[np.newaxis]; 


        # gradient descent parameters
        self.eps = 10**(-5);            # termination condition: exit gradient descent loop 
                                        # if J changes less than eps in one step
        self.n_steps = 0;               # counter for steps in gradient descent loop
        self.max_steps = 300;           # number of steps after which loop is terminated
        self.beta_opt = 10**(-4);              # gradient step width parameter initialization
        self.beta_mult_phase_gradadd = 10**(-4);
        self.beta_mult_phase_gradopt = 10**(-4);
        self.beta_rel = 0.04;           # gradient step relative accuracy
        self.mult_tol = 0.5;            # multiplicity tolerance for n-th eigenvalue, 
                                        # all eigenvalues in [lambda_n-mult_tol , lambda_n] 
                                        # are interpreted as one eigenvalue with multiplicity > 1,
                                        # so far only relevant when sum_length = 1

        # set parameters for genetic algorithm
        self.N_best = 20;                    # no. of fittest domains to keep in each step
        self.N_rand = 20;                    # no. of newly added random domains in each step
        self.N_pop  = 200;                   # population size
        self.N_iter = 10;                    # number of iterations
        
        if self.p0 == 1:
            p_string = ""
        else: 
            p_string = "p" + str(self.p0)
        if self.sum_length == 1:
            self.file_out_name = "lambda" + str(self.n) + p_string + "B" + str(self.B0)   
        else:
            self.file_out_name = "sum" + str(self.n) + p_string + "B" + str(self.B0)
        self.file_out_name_ga = "ga_" + self.file_out_name
        self.file_out_name_r = "r_" + self.file_out_name
        self.verbose = 1

        # set parameters for random optimization algorithm
        self.dev_max = 0.01

        if not(self.operator == "DL" or self.operator == "MDL"):
            print("Error: invalid operator type")
            exit()

        
    def set_sum_length(self, new_sum_length):
        self.sum_length = new_sum_length
                                  
# Define domain class
# computes important variables that describe the domain: variable radius and its derivative,
# collocation points, normal vectors, source points, area and center of mass
class Domain:
    def __init__(self,c_,param_): 
        self.param = param_
        self.c = c_                     # row-wise coefficient vectors
        self.a = c_[0:1,:]
        self.b = c_[1:2,:]
        self.Gamma_col = np.zeros((2,self.param.N_col))
        self.Normal_temp = np.zeros((2,self.param.N_col))
        self.Normal_norm = np.zeros((1,self.param.N_col))
        self.Normal_col = np.zeros((2,self.param.N_col))
        self.p_source = np.zeros((2,self.param.N_col))
        self.CoM = np.zeros((2,))

        # compute area, center of mass and curve increment for curve integrals
        f = lambda t: 0.5*self.r(t)**2
        self.A_Gamma = self.integrate_over_boundary(f)
        f = lambda t: 1.0/(3.0*self.A_Gamma)*self.r(t)**3*np.cos(t)
        self.CoM[0] = self.integrate_over_boundary(f)
        f = lambda t: 1.0/(3.0*self.A_Gamma)*self.r(t)**3*np.sin(t)
        self.CoM[1] = self.integrate_over_boundary(f)

        self.dbdy_arc_len = 1           # apply set_t_col before doing curve integrals!!!

    def rescale_domain(self):
        self.c = self.c/np.sqrt(self.A_Gamma)
        self.a = self.c[0:1,:]
        self.b = self.c[1:2,:]
        
        # recompute area, center of mass
        f = lambda t: 0.5*self.r(t)**2
        self.A_Gamma = self.integrate_over_boundary(f)
        f = lambda t: 1.0/(3.0*self.A_Gamma)*self.r(t)**3*np.cos(t)
        self.CoM[0] = self.integrate_over_boundary(f)
        f = lambda t: 1.0/(3.0*self.A_Gamma)*self.r(t)**3*np.sin(t)
        self.CoM[1] = self.integrate_over_boundary(f)

    def compute_inertia_principal_axes(self):
        I = np.zeros((2,2))     # inertia tensor

        # compute tensor elements
        f = lambda t: (self.r(t)**4/4.0) * np.sin(t)**2 
        I[0,0] = self.integrate_over_boundary(f)

        f = lambda t: (self.r(t)**4/4.0) * np.cos(t)**2 
        I[1,1] = self.integrate_over_boundary(f)

        f = lambda t: - (self.r(t)**4/4.0) * np.sin(t)*np.cos(t) 
        I[0,1] = self.integrate_over_boundary(f)
        I[1,0] = I[0,1]

        # shift to center of mass
        I[0,0] = I[0,0] - self.A_Gamma * self.CoM[1]**2            # parallel axis theorem (mass = area here)
        I[1,1] = I[1,1] - self.A_Gamma * self.CoM[0]**2   
        I[0,1] = I[0,1] - self.A_Gamma * self.CoM[0]*self.CoM[1]
        I[1,0] = I[0,1]

        _, p_axis = np.linalg.eigh(I)
        theta_angle = 0.5 * np.arctan(2*I[0,1]/(I[1,1]-I[0,0]))

        return I, p_axis, theta_angle

    # distribute collocation points evenly over boundary
    def set_t_col_init(self):        
        
        # arc length at t_int points
        bdy_arc_len_prev = np.zeros(np.shape(self.param.t_int))
        
        # integrate arc length
        for k in range(1,self.param.N_int):
            bdy_arc_len_prev[0,k] = bdy_arc_len_prev[0,k-1] + self.param.dt_int * np.linalg.norm(self.dt_gamma_xy(self.param.t_int[0,k-1]))
                    
        self.dbdy_arc_len = bdy_arc_len_prev[0,-1]/self.param.N_col
        bdy_arc_len_want = np.linspace(0,bdy_arc_len_prev[0,-1] - self.dbdy_arc_len,self.param.N_col)[np.newaxis]
            
        t_new = np.zeros(np.shape(self.param.t_col))
                
        for k in range(1,self.param.N_col):
            ind = np.searchsorted(bdy_arc_len_prev[0,:] , bdy_arc_len_want[0,k] )
            t_new[0,k] = (bdy_arc_len_want[0,k]-bdy_arc_len_prev[0,ind-1])/(bdy_arc_len_prev[0,ind]-bdy_arc_len_prev[0,ind-1])*(self.param.t_int[0,ind]-self.param.t_int[0,ind-1]) + self.param.t_int[0,ind-1]
                
        return t_new         # overwrites self.param !!
    
    # set t_col, optimize if wanted
    def set_t_col(self, t_init=None, do_node_repel=False, node_repel_n_max_iter=0):
        if t_init is None:
            # make initial guess for collocation points on boundary

            # approx. normals
            t_init_         = self.set_t_col_init()
                           
        # apply node repel algorithm to optimize collocation points if wanted
        if do_node_repel:
            self.param.t_col = self.node_repel(t_init_, self.param.node_repel_p, node_repel_n_max_iter) 
        else:
            self.param.t_col = t_init_
            
    # node repel optimization algorithm
    # fixes points at t=0 and let other points move
    def node_repel(self, t_init, p, n_max_iter):
    
        # fix angles at t = 0
        t_fixed = t_init[0,0]
        angles_init  = t_init[0,1:]       # remaining angels that will be optimized
        n_angles = len(angles_init)

        def f(angles):
            
            t_temp = np.hstack( (t_fixed,angles[:n_angles]) )[np.newaxis]
            
            Gamma_temp = self.gamma_xy(t_temp)
            diff_matrix_x = np.subtract.outer( Gamma_temp[0,:], Gamma_temp[0,:] );
            diff_matrix_y = np.subtract.outer( Gamma_temp[1,:], Gamma_temp[1,:] );
            diff_xy = np.sqrt(diff_matrix_x**2+diff_matrix_y**2);
            
            E_matrix = 1/diff_xy**p                 
            E_matrix[np.isnan(E_matrix)] = 0.0

            E = sum( [np.sum(E_matrix[i][:i]) for i in range(n_angles) ] )    #total energy

            return E
        
        def gradient_f(angles):
            
            t_temp = np.hstack( (t_fixed,angles[:n_angles]) )[np.newaxis]
            
            Gamma_temp = self.gamma_xy(t_temp)
            dt_Gamma_temp = self.dt_gamma_xy(t_temp)
            
            diff_matrix_x = np.subtract.outer( Gamma_temp[0,:], Gamma_temp[0,:] );
            diff_matrix_y = np.subtract.outer( Gamma_temp[1,:], Gamma_temp[1,:] );
            diff_xy = np.sqrt(diff_matrix_x**2+diff_matrix_y**2);
            
            E_matrix = 1/diff_xy**(p+2)               
            E_matrix[np.isnan(E_matrix)] = 0.0
            
            dt_Gamma_temp_matrix_x = np.multiply.outer( dt_Gamma_temp[0,:], np.ones(np.shape(Gamma_temp[0,:])) )
            dt_Gamma_temp_matrix_y = np.multiply.outer( dt_Gamma_temp[1,:], np.ones(np.shape(Gamma_temp[1,:])) )
            
            # gradient in t-direction
            dt_E = -p*E_matrix * (diff_matrix_x * dt_Gamma_temp_matrix_x + diff_matrix_y * dt_Gamma_temp_matrix_y)
            dt_E[np.isnan(dt_E)] = 0.0
            dt_E = np.sum(dt_E, axis=1)
            
            gradient = dt_E[1:]

            return gradient
        
        angles_opt = scipy.optimize.fmin_bfgs(f, angles_init, fprime=gradient_f, gtol=1e-05, maxiter=n_max_iter)
        
        t_opt = np.hstack( (t_fixed,angles_opt) ) 
        
        return t_opt[np.newaxis]

    # compute interior points for SAT
    def compute_interior_points(self, N):
        Gamma_interior = np.zeros((2,N))
        i = 0
        while i < N:
            rand_point = 1.0-2.0*np.random.rand(2,50)
            X_ = rand_point[0,:]; Y_ = rand_point[1,:];
            
            R_, T_ = self.r_t_from_xy(X_, Y_)        # get angles from xyz coords

            admis_rand_point = rand_point[:,R_ < 0.95*self.r(T_)]  # check if point is admissible (interior point)
            
            for k in range(np.shape(admis_rand_point)[1]):
                if i < N:
                    Gamma_interior[:,i] = admis_rand_point[:,k]
                    i = i + 1
            
        return Gamma_interior

    def set_domain_variables(self):        
        
        # collocation points on domain boundary
        self.Gamma_col =  self.gamma_xy(self.param.t_col)
        self.Gamma_interior = self.compute_interior_points(self.param.N_interior)

        # normal vectors in collocation points
        self.Normal_temp = np.vstack((self.dt_gamma_y(self.param.t_col),-self.dt_gamma_x(self.param.t_col)))
        self.Normal_norm = np.sqrt(self.Normal_temp[0,:]**2+ self.Normal_temp[1,:]**2)
        self.Normal_temp[0,:] = self.Normal_temp[0,:]/self.Normal_norm
        self.Normal_temp[1,:] = self.Normal_temp[1,:]/self.Normal_norm
        self.Normal_col = self.Normal_temp
        
        # source points with approx. normals
        Gamma_shiftp1 = np.zeros(np.shape(self.Gamma_col))
        Gamma_shiftm1 = np.zeros(np.shape(self.Gamma_col))
        Gamma_shiftp1[:,0:-1] = self.Gamma_col[:,1:]
        Gamma_shiftp1[:,-1] = self.Gamma_col[:,0]
        Gamma_shiftm1[:,1:] = self.Gamma_col[:,0:-1]
        Gamma_shiftm1[:,0] = self.Gamma_col[:,-1]
        p_temp = np.zeros(np.shape(self.Gamma_col))
        p_temp[0,:] = 0.5 * (Gamma_shiftp1[1,:] - Gamma_shiftm1[1,:]); 
        p_temp[1,:] = -0.5 * (Gamma_shiftp1[0,:] - Gamma_shiftm1[0,:]); 
        p_norm = np.sqrt(p_temp[0,:]**2+ p_temp[1,:]**2);
        p_temp[0,:] = p_temp[0,:]/p_norm
        p_temp[1,:] = p_temp[1,:]/p_norm

        self.p_source = self.Gamma_col + self.param.p_beta * p_temp    



              
    def integrate_over_boundary(self, f):
        # f must be able to evaluate arrays of angles t
        # integrate over t, add line weight to f when necessary
        
        integral = self.param.dt_int * np.sum( f(self.param.t_int) )   

        return integral

    # Define radius function
    # defines the variable radius and its derivative for given Fourier
    # coefficients

    def r(self,t):
        r = np.dot(np.cos(np.multiply.outer(t,np.linspace(0,self.param.len_ab-1,self.param.len_ab))),self.a[0,:]) +  np.dot(np.sin(np.multiply.outer(t,np.linspace(0,self.param.len_ab-1,self.param.len_ab))),self.b[0,:])
        return r 
      
    def dr_dt(self,t):      
        dr_dt = (  -np.dot(np.sin(np.kron(np.linspace(0,self.param.len_ab-1,self.param.len_ab),t.T)),(np.linspace(0,self.param.len_ab-1,self.param.len_ab)*self.a).T)  +   np.dot(np.cos(np.kron(np.linspace(0,self.param.len_ab-1,self.param.len_ab),t.T)),(np.linspace(0,self.param.len_ab-1,self.param.len_ab)*self.b).T)  ).T 
        return dr_dt

    def r_t_from_xy(self, X_, Y_ ):
        R_ = np.sqrt(X_**2+Y_**2)
        T_ = np.arccos(X_/R_)
        T_[Y_<0] = -T_[Y_<0] 
        return R_, T_

    def outside_domain(self, X_, Y_ ): 
        R, T = self.r_t_from_xy(X_, Y_ )   
        R = np.reshape(R, (np.product(np.shape(X_)),) )
        T = np.reshape(T, (np.product(np.shape(X_)),) )
        ind_comp = np.greater(R, self.r(T))                  # point indices where computation is not necessary (points outside domain)
        ind_comp = np.reshape( ind_comp, np.shape(X_) )
        return ind_comp

    def get_c(self):
        return self.c

    def gamma_xy(self, t):
        r_ = self.r(t)
        x = r_*np.cos(t)
        y = r_*np.sin(t)

        gamma_xy_ = np.vstack((x,y))

        return gamma_xy_

    def dt_gamma_x(self,t):      
        dg_x_dt = self.dr_dt(t) * np.cos(t) - self.r(t) * np.sin(t)
        return dg_x_dt

    def dt_gamma_y(self,t):      
        dg_y_dt = self.dr_dt(t) * np.sin(t) + self.r(t) * np.cos(t)
        return dg_y_dt

    def dt_gamma_xy(self, t):
        x = self.dt_gamma_x(t)
        y = self.dt_gamma_y(t)
        dt_gamma_xy_ = np.vstack((x,y))

        return dt_gamma_xy_


class GeneralizedDomain(Domain):
    def __init__(self,c_,param_): 
        self.param = param_
        self.c = c_                 # row-wise coefficient vectors
        self.a_x = c_[0:1,:]
        self.b_x = c_[1:2,:]
        self.a_y = c_[2:3,:]
        self.b_y = c_[3:4,:]
        self.Gamma_col = np.zeros((2,self.param.N_col))
        self.Normal_temp = np.zeros((2,self.param.N_col))
        self.Normal_norm = np.zeros((1,self.param.N_col))
        self.Normal_col = np.zeros((2,self.param.N_col))
        self.p_source = np.zeros((2,self.param.N_col))
        self.CoM = np.zeros((2,))

        # compute area, center of mass and curve increment for curve integrals
        f = lambda t: 0.5*(self.gamma_x(t) * self.dt_gamma_y(t) - self.gamma_y(t) * self.dt_gamma_x(t))
        self.A_Gamma = self.integrate_over_boundary(f)

        if self.A_Gamma < 0:                    # if boundary curve is not going mathematically positive way, the flip sine coefficients to flip direction of curve
            self.c[1:2,:] = -self.c[1:2,:]
            self.c[3:4,:] = -self.c[3:4,:]
            self.b_x = c_[1:2,:]
            self.b_y = c_[3:4,:]
            self.A_Gamma = -self.A_Gamma

        f = lambda t: 0.25 * self.gamma_x(t)**2 * self.dt_gamma_y(t)                    - 0.5 * self.gamma_x(t) * self.gamma_y(t) * self.dt_gamma_x(t)
        self.CoM[0] = self.integrate_over_boundary(f) / self.A_Gamma
        f = lambda t: 0.5 * self.gamma_x(t) * self.gamma_y(t) * self.dt_gamma_y(t)    - 0.25 * self.gamma_y(t)**2 * self.dt_gamma_x(t)
        self.CoM[1] = self.integrate_over_boundary(f) / self.A_Gamma
        
        self.dbdy_arc_len = 1           # apply set_t_col before doing curve integrals!!!

    def rescale_domain(self):
        self.c = self.c/np.sqrt(self.A_Gamma)
        self.a_x = self.c[0:1,:]
        self.b_x = self.c[1:2,:]
        self.a_y = self.c[2:3,:]
        self.b_y = self.c[3:4,:]
        
        # recompute area, center of mass 
        f = lambda t: 0.5*(self.gamma_x(t) * self.dt_gamma_y(t) - self.gamma_y(t) * self.dt_gamma_x(t))
        self.A_Gamma = self.integrate_over_boundary(f)
        f = lambda t: 0.25 * self.gamma_x(t)**2 * self.dt_gamma_y(t)                    - 0.5 * self.gamma_x(t) * self.gamma_y(t) * self.dt_gamma_x(t)
        self.CoM[0] = self.integrate_over_boundary(f) / self.A_Gamma
        f = lambda t: 0.5 * self.gamma_x(t) * self.gamma_y(t) * self.dt_gamma_y(t)    - 0.25 * self.gamma_y(t)**2 * self.dt_gamma_x(t)
        self.CoM[1] = self.integrate_over_boundary(f) / self.A_Gamma
    
    # generates domain with de-curled c coefficients by regression with (basically enforced) arc-length parametrization 
    def decurl_coeff(self, len_ab_new):
        
        def gamma_x_temp(t,a_x,b_x,a_y,b_y):
            g_x = np.dot(np.cos(np.multiply.outer(t,np.linspace(0,len_ab_new-1,len_ab_new))),a_x[0,:]) +  np.dot(np.sin(np.multiply.outer(t,np.linspace(0,len_ab_new-1,len_ab_new))),b_x[0,:])
            return g_x 

        def gamma_y_temp(t,a_x,b_x,a_y,b_y):
            g_y = np.dot(np.cos(np.multiply.outer(t,np.linspace(0,len_ab_new-1,len_ab_new))),a_y[0,:]) +  np.dot(np.sin(np.multiply.outer(t,np.linspace(0,len_ab_new-1,len_ab_new))),b_y[0,:])
            return g_y 
        
        def dt_gamma_x_temp(t,a_x,b_x,a_y,b_y):      
            dg_x_dt = (  -np.dot(np.sin(np.kron(np.linspace(0,len_ab_new-1,len_ab_new),t.T)),(np.linspace(0,len_ab_new-1,len_ab_new)*a_x).T)  +   np.dot(np.cos(np.kron(np.linspace(0,len_ab_new-1,len_ab_new),t.T)),(np.linspace(0,len_ab_new-1,len_ab_new)*b_x).T)  ).T 
            return dg_x_dt

        def dt_gamma_y_temp(t,a_x,b_x,a_y,b_y):      
            dg_y_dt = (  -np.dot(np.sin(np.kron(np.linspace(0,len_ab_new-1,len_ab_new),t.T)),(np.linspace(0,len_ab_new-1,len_ab_new)*a_y).T)  +   np.dot(np.cos(np.kron(np.linspace(0,len_ab_new-1,len_ab_new),t.T)),(np.linspace(0,len_ab_new-1,len_ab_new)*b_y).T)  ).T 
            return dg_y_dt

        c_ = self.get_c()
        shape_c = np.shape(c_)

        t_equi = np.linspace(0,(2*np.pi-self.param.dt_col[0]),self.param.N_col)[np.newaxis]; 
        
        def f(x):
            c = x.reshape((shape_c[0],len_ab_new)) 
            a_x = c[0:1,:]
            b_x = c[1:2,:]
            a_y = c[2:3,:]
            b_y = c[3:4,:]
            sum2 = np.sum( (self.Gamma_col[0,:] -self.CoM[0] - gamma_x_temp(t_equi,a_x,b_x,a_y,b_y))**2 +  ( self.Gamma_col[1,:] -self.CoM[1] - gamma_y_temp(t_equi,a_x,b_x,a_y,b_y))**2 ) #+ 10000*np.sum( (dt_gamma_x_temp(t_equi,c)**2 +  dt_gamma_y_temp(t_equi,c)**2 - 1)**2 )
            return sum2

        c_ = c_[..., :len_ab_new]
        x0 = c_.reshape((shape_c[0]*len_ab_new,)) 
        
        res = scipy.optimize.minimize(f, x0, tol=1e-6)

        c_decurl = np.array(res.x)[np.newaxis]
        c_decurl = c_decurl.reshape((shape_c[0],len_ab_new)) 
        c_decurl = np.hstack((c_decurl, np.zeros((shape_c[0], self.param.len_ab - len_ab_new)))) 
        
        return c_decurl

    def compute_interior_points(self, N): 
        Gamma_interior = np.zeros((2,N))
        i = 0
        while i < N:
            rand_point = 1.0-2.0*np.random.rand(2,20)
            X_ = rand_point[0,:]; Y_ = rand_point[1,:];
            
            ind_comp = self.outside_domain(X_, Y_ )     # check if point is not admissible (exterior point)
            ind_comp = np.logical_not(ind_comp)         # flip truth value to get interior points
            
            admis_rand_point = rand_point[:,ind_comp]   # only points in domain are admissible

            for k in range(np.shape(admis_rand_point)[1]):
                if i < N:
                    Gamma_interior[:,i] = admis_rand_point[:,k]
                    i = i + 1
                else:
                    break
            
        return Gamma_interior

    def set_domain_variables(self):        

        # collocation points on domain boundary
        self.Gamma_col =  self.gamma_xy(self.param.t_col)
        self.Gamma_interior = self.compute_interior_points(self.param.N_interior)

        # normal vectors in collocation points
        self.Normal_temp = np.vstack((self.dt_gamma_y(self.param.t_col),-self.dt_gamma_x(self.param.t_col)))
        self.Normal_norm = np.sqrt(self.Normal_temp[0,:]**2+ self.Normal_temp[1,:]**2);
        self.Normal_temp[0,:] = self.Normal_temp[0,:]/self.Normal_norm
        self.Normal_temp[1,:] = self.Normal_temp[1,:]/self.Normal_norm
        self.Normal_col = self.Normal_temp
        
        # source points with approx. normals
        Gamma_shiftp1 = np.zeros(np.shape(self.Gamma_col))
        Gamma_shiftm1 = np.zeros(np.shape(self.Gamma_col))
        Gamma_shiftp1[:,0:-1] = self.Gamma_col[:,1:]
        Gamma_shiftp1[:,-1] = self.Gamma_col[:,0]
        Gamma_shiftm1[:,1:] = self.Gamma_col[:,0:-1]
        Gamma_shiftm1[:,0] = self.Gamma_col[:,-1]
        p_temp = np.zeros(np.shape(self.Gamma_col))
        p_temp[0,:] = 0.5 * (Gamma_shiftp1[1,:] - Gamma_shiftm1[1,:]); 
        p_temp[1,:] = -0.5 * (Gamma_shiftp1[0,:] - Gamma_shiftm1[0,:]); 
        p_norm = np.sqrt(p_temp[0,:]**2+ p_temp[1,:]**2);
        p_temp[0,:] = p_temp[0,:]/p_norm
        p_temp[1,:] = p_temp[1,:]/p_norm
        
        rho = 0.3
        dt_Gamma_col =  self.dt_gamma_xy(self.param.t_col)
        dt2_Gamma_col =  self.dt2_gamma_xy(self.param.t_col)
        curvature = (dt_Gamma_col[0,:] * dt2_Gamma_col[1,:] - dt_Gamma_col[1,:] * dt2_Gamma_col[0,:]) / (dt_Gamma_col[0,:]** 2 + dt_Gamma_col[1,:]**2 ) ** (1.5)
        curvature[curvature>0] =0.0
                
        #p_beta_mult = 0.1 + 0.9 / (1.0 + rho * np.abs(curvature)**0.5) # /( 1.0 + rho * ( (np.mean(curvature)-curvature)/(np.mean(curvature)-np.min(curvature)) ) )
        p_beta_mult = 1.0
        
        #plt.plot(curvature, 'b')
        #plt.plot(p_beta_mult, 'g')
        #plt.show()
           
        #p_source_temp = self.Gamma_col + self.param.p_beta * p_beta_mult * p_temp
        #self.N_p_source = np.shape(p_source_temp)[1]
        #p_source_temp_mean_dist = np.zeros((self.N_p_source,))
        #        
        #diff_matrix_x = np.subtract.outer( p_source_temp[0,:], p_source_temp[0,:])
        #diff_matrix_y = np.subtract.outer( p_source_temp[1,:], p_source_temp[1,:])
        #dist_matrix = np.sqrt(diff_matrix_x**2+diff_matrix_y**2)     
        #       
        #for k in range(self.N_p_source):
        #    p_source_temp_mean_dist[k] = dist_matrix[k,k-1]
        #p_source_temp_mean_dist = np.mean(p_source_temp_mean_dist)
        # 
        ### test: delete source points that are too close -> worse quality ###
        #k = 1
        #self.p_source = p_source_temp[:,0:1]
        #while k < np.shape(dist_matrix)[0]:
        #    if np.min(dist_matrix[k,:k]) < 0.0 * p_source_temp_mean_dist: 
        #        p_source_temp = np.delete(p_source_temp, [k], axis=1)
        #        diff_matrix_x = np.subtract.outer( p_source_temp[0,:], p_source_temp[0,:])
        #        diff_matrix_y = np.subtract.outer( p_source_temp[1,:], p_source_temp[1,:])
        #        dist_matrix = np.sqrt(diff_matrix_x**2+diff_matrix_y**2)
        #    else:
        #        self.p_source = np.hstack((self.p_source, p_source_temp[:,k:k+1]))
        #        k = k+1
        #
        ### test: take averaged source points -> worse quality###
        #p_source_act = p_source_temp[:,0:1]
        #self.p_source = p_source_act
        #for k in range(1,self.N_p_source):
        ##   print(p_source_act)
        #   print(p_source_temp[:,k:k+1])
        #   
        #   if np.linalg.norm(p_source_temp[:,k:k+1]-np.mean(p_source_act,axis=1)) < 0.5 * p_source_temp_mean_dist:
        #      p_source_act = np.vstack((p_source_act, p_source_temp[k:k+1]))
        ##  else:
        #      self.p_source = np.hstack((self.p_source, p_source_act))
        #      p_source_act = p_source_temp[:,k:k+1]
            
        
        
        self.p_source = self.Gamma_col + self.param.p_beta * p_beta_mult * p_temp
        self.N_p_source = np.shape(self.p_source)[1]
    
    def integrate_over_boundary(self, f):
        # f must be able to evaluate arrays of angles t
        # integrate over t, add line weight to f when necessary

        integral = self.param.dt_int * np.sum( f(self.param.t_int) )   

        return integral

    def integrate_over_boundary_matrix(self, f, direction=0):
        # f must be able to evaluate matrixes of angles t
        # integrate over t, add line weight to f when necessary

        integral = self.param.dt_int * np.sum( f(self.param.t_int), axis=direction )   

        return integral

    # Define boundary curve function gamma
    # defines the boundary curve and its derivatives for given Fourier
    # coefficients

    def gamma_x(self,t):
        g_x = np.dot(np.cos(np.multiply.outer(t,np.linspace(0,self.param.len_ab-1,self.param.len_ab))),self.a_x[0,:]) +  np.dot(np.sin(np.multiply.outer(t,np.linspace(0,self.param.len_ab-1,self.param.len_ab))),self.b_x[0,:])
        return g_x 

    def gamma_y(self,t):
        g_y = np.dot(np.cos(np.multiply.outer(t,np.linspace(0,self.param.len_ab-1,self.param.len_ab))),self.a_y[0,:]) +  np.dot(np.sin(np.multiply.outer(t,np.linspace(0,self.param.len_ab-1,self.param.len_ab))),self.b_y[0,:])
        return g_y 

    def dt_gamma_x(self,t):      
        dg_x_dt = (  -np.dot(np.sin(np.kron(np.linspace(0,self.param.len_ab-1,self.param.len_ab),t.T)),(np.linspace(0,self.param.len_ab-1,self.param.len_ab)*self.a_x).T)  +   np.dot(np.cos(np.kron(np.linspace(0,self.param.len_ab-1,self.param.len_ab),t.T)),(np.linspace(0,self.param.len_ab-1,self.param.len_ab)*self.b_x).T)  ).T 
        return dg_x_dt

    def dt_gamma_y(self,t):      
        dg_y_dt = (  -np.dot(np.sin(np.kron(np.linspace(0,self.param.len_ab-1,self.param.len_ab),t.T)),(np.linspace(0,self.param.len_ab-1,self.param.len_ab)*self.a_y).T)  +   np.dot(np.cos(np.kron(np.linspace(0,self.param.len_ab-1,self.param.len_ab),t.T)),(np.linspace(0,self.param.len_ab-1,self.param.len_ab)*self.b_y).T)  ).T 
        return dg_y_dt
    
    def dt2_gamma_x(self,t):      
        dg_x_dt = (  -np.dot(np.cos(np.kron(np.linspace(0,self.param.len_ab-1,self.param.len_ab),t.T)),(np.linspace(0,self.param.len_ab-1,self.param.len_ab)*np.linspace(-1,self.param.len_ab-2,self.param.len_ab)*self.a_x).T)  -   np.dot(np.sin(np.kron(np.linspace(0,self.param.len_ab-1,self.param.len_ab),t.T)),(np.linspace(0,self.param.len_ab-1,self.param.len_ab)*np.linspace(-1,self.param.len_ab-2,self.param.len_ab)*self.b_x).T)  ).T 
        return dg_x_dt

    def dt2_gamma_y(self,t):      
        dg_y_dt = (  -np.dot(np.cos(np.kron(np.linspace(0,self.param.len_ab-1,self.param.len_ab),t.T)),(np.linspace(0,self.param.len_ab-1,self.param.len_ab)*np.linspace(-1,self.param.len_ab-2,self.param.len_ab)*self.a_y).T)  -   np.dot(np.sin(np.kron(np.linspace(0,self.param.len_ab-1,self.param.len_ab),t.T)),(np.linspace(0,self.param.len_ab-1,self.param.len_ab)*np.linspace(-1,self.param.len_ab-2,self.param.len_ab)*self.b_y).T)  ).T 
        return dg_y_dt

    # check if point is in domain by computing the winding number
    def outside_domain(self, X_, Y_ ): 
        f = lambda t: (np.subtract.outer(self.gamma_x(t), X_) * np.multiply.outer(self.dt_gamma_y(t), np.ones(np.shape(Y_))) - np.subtract.outer(self.gamma_y(t), Y_) * np.multiply.outer(self.dt_gamma_x(t), np.ones(np.shape(X_)))) / (np.subtract.outer(self.gamma_x(t), X_)**2 + np.subtract.outer(self.gamma_y(t), Y_)**2 )
        winding_number = self.integrate_over_boundary_matrix(f, direction=(0,1)) / (2.0*np.pi)

        ind_comp = np.less(np.abs(winding_number), 10**(-2))
        
        #for k in range(np.shape(X_)[0]):
        #    for l in range(np.shape(X_)[1]):
        #        if ind_comp:
        #plt.plot(X_, Y_,'k.')
        #plt.plot(X_[ind_comp], Y_[ind_comp],'r.')
        #plt.show()
        
        return ind_comp

    def get_c(self):
        return self.c

    def gamma_xy(self, t):
        x = self.gamma_x(t)
        y = self.gamma_y(t)
        gamma_xy_ = np.vstack((x,y))

        return gamma_xy_

    def dt_gamma_xy(self, t):
        x = self.dt_gamma_x(t)
        y = self.dt_gamma_y(t)
        dt_gamma_xy_ = np.vstack((x,y))

        return dt_gamma_xy_
    
    def dt2_gamma_xy(self, t):
        x = self.dt2_gamma_x(t)
        y = self.dt2_gamma_y(t)
        dt2_gamma_xy_ = np.vstack((x,y))

        return dt2_gamma_xy_



# rectangular domain class, only for testing MFS, no optimization supported

class Rectangle(Domain):
    def __init__(self,side_a,side_b,param_): 
        self.param = param_
        self.side_a = side_a
        self.side_b = side_b
        self.N_side_a = int(self.side_a/(self.side_a+self.side_b) * self.param.N_col/2)-1        # number of on side_a points 
        self.N_side_b = int(self.side_b/(self.side_a+self.side_b) * self.param.N_col/2)-1        # number of on side_b points, 2*N_side_a+2*N_side_b ~= N_col 
        self.param.N_col = 2*self.N_side_a + 2*self.N_side_b 
        self.Gamma_col = np.zeros((2,self.param.N_col))
        self.Normal_col = np.zeros((2,self.param.N_col))
        self.p_source = np.zeros((2,self.param.N_col))
        self.CoM = np.zeros((2,))

        # compute area, center of mass is origin
        self.A_Gamma = self.side_a * self.side_b

        self.dbdy_arc_len = 1           # apply set_t_col before doing curve integrals!!!

    def rescale_domain(self):
        self.side_a = self.side_a/np.sqrt(self.A_Gamma)
        self.side_b = self.side_b/np.sqrt(self.A_Gamma)
        
        # recompute area
        self.A_Gamma = self.side_a * self.side_b

    # distribute collocation points evenly over boundary
    def set_t_col(self):        
        return None #empty

    def compute_interior_points(self, N):
        Gamma_interior = np.zeros((2,N))

        rand_point = 0.5-np.random.rand(2,N)
        Gamma_interior[0,:] = self.side_a * rand_point[0,:]
        Gamma_interior[1,:] = self.side_b * rand_point[1,:]

        return Gamma_interior

    def set_domain_variables(self):       
        
        
        
        # collocation points on domain boundary

        #equidistant spacing
        legg_a = np.linspace(-1,1,num = self.N_side_a+1); legg_a = legg_a[1:-1]
        legg_b = np.linspace(-1,1,num = self.N_side_b+1); legg_b = legg_b[1:-1]
        
        #leggauss spacing (worse than equidistant)
        #legg_a, _ = np.polynomial.legendre.leggauss(self.N_side_a-1)    # does not include corners!
        #legg_b, _ = np.polynomial.legendre.leggauss(self.N_side_b-1)
        #legg_a = np.sign(legg_a) * np.abs(legg_a)**1.3
        #legg_b = np.sign(legg_b) * np.abs(legg_b)**1.3
        self.Gamma_col[0,1:self.N_side_a] = 0.5*self.side_a * legg_a        #oben
        self.Gamma_col[1,1:self.N_side_a] = 0.5*self.side_b
        self.Gamma_col[0,0] = -0.5*self.side_a                              #Ecke oben links
        self.Gamma_col[1,0] = 0.5*self.side_b
        self.Gamma_col[0,self.N_side_a+1:2*self.N_side_a] = 0.5*self.side_a * legg_a
        self.Gamma_col[1,self.N_side_a+1:2*self.N_side_a] = -0.5*self.side_b
        self.Gamma_col[0,self.N_side_a] = 0.5*self.side_a                              #Ecke unten rechts
        self.Gamma_col[1,self.N_side_a] = -0.5*self.side_b
        self.Gamma_col[0,2*self.N_side_a+1:2*self.N_side_a+self.N_side_b] = 0.5*self.side_a
        self.Gamma_col[1,2*self.N_side_a+1:2*self.N_side_a+self.N_side_b] = 0.5*self.side_b * legg_b
        self.Gamma_col[0,2*self.N_side_a] = 0.5*self.side_a                              #Ecke oben rechts
        self.Gamma_col[1,2*self.N_side_a] = 0.5*self.side_b
        self.Gamma_col[0,2*self.N_side_a+self.N_side_b+1:2*self.N_side_a+2*self.N_side_b] = -0.5*self.side_a
        self.Gamma_col[1,2*self.N_side_a+self.N_side_b+1:2*self.N_side_a+2*self.N_side_b] = 0.5*self.side_b * legg_b
        self.Gamma_col[0,2*self.N_side_a+self.N_side_b] = -0.5*self.side_a                              #Ecke unten links
        self.Gamma_col[1,2*self.N_side_a+self.N_side_b] = -0.5*self.side_b
        
        
        
        self.Gamma_interior = self.compute_interior_points(self.param.N_interior)

        # normal vectors in collocation points
        self.Normal_col[0,0:self.N_side_a] = 0.0    #oben
        self.Normal_col[1,0:self.N_side_a] = 1.0
        self.Normal_col[0,self.N_side_a:2*self.N_side_a] = 0.0    #unten
        self.Normal_col[1,self.N_side_a:2*self.N_side_a] = -1.0
        self.Normal_col[0,2*self.N_side_a:2*self.N_side_a+self.N_side_b] = 1.0    #rechts
        self.Normal_col[1,2*self.N_side_a:2*self.N_side_a+self.N_side_b] = 0.0
        self.Normal_col[0,2*self.N_side_a+self.N_side_b:2*self.N_side_a+2*self.N_side_b] = -1.0    #links
        self.Normal_col[1,2*self.N_side_a+self.N_side_b:2*self.N_side_a+2*self.N_side_b] = 0.0
        
        # source points with approx. normals
        p_temp = self.Normal_col
        p_temp[0,0] = -1.0/np.sqrt(2)
        p_temp[1,0] = 1.0/np.sqrt(2)
        p_temp[0,self.N_side_a] = 1.0/np.sqrt(2)
        p_temp[1,self.N_side_a] = -1.0/np.sqrt(2)
        p_temp[0,2*self.N_side_a] = 1.0/np.sqrt(2)
        p_temp[1,2*self.N_side_a] = 1.0/np.sqrt(2)
        p_temp[0,2*self.N_side_a+self.N_side_b] = -1.0/np.sqrt(2)
        p_temp[1,2*self.N_side_a+self.N_side_b] = -1.0/np.sqrt(2)
        self.p_source = self.Gamma_col + self.param.p_beta * self.Normal_col#p_temp    
        self.N_p_source = np.shape(self.p_source)[1]
              
    def integrate_over_boundary(self, f):
        return None #empty

    def r(self,t):
        return None #empty
      
    def dr_dt(self,t):      
        return None #empty
        
    def r_t_from_xy(self, X_, Y_ ):
        return None #empty

    def get_ab(self):
        return None #empty

    def gamma_xy(self, t):
        return None #empty

    def dt_gamma_xy(self, t):
        return None #empty





# Optimize coefficients by applying random deviations
#
def random_optimization(filename_ab, filename_val, lineno, param):
    c_in = read_from_file(filename_ab, lineno)
    a,_    = read_from_file_val(filename_val, lineno); 
    objJ = a
    c_shape = np.shape(c_in)
    
    print("base objective: " + str(objJ))
    
    while param.n_steps < param.max_steps:
        
        time_start = time.time()
        
        # randomize coefficients
        c_temp = (1+param.dev_max-2*param.dev_max*rand(c_shape[0],c_shape[1]))*c_in
        
        domain = GeneralizedDomain(c_temp, param)
        domain.rescale_domain()               # rescale and redistribute collocation points
        domain.set_t_col(t_init=None, do_node_repel=domain.param.do_node_repel, node_repel_n_max_iter=domain.param.node_repel_n_max_iter)               # redefine t_col points for equi.arc.dist
        domain.set_domain_variables()         # set collocation and source points
        objJ_temp, E_crit_temp, _, _, _ = direct_problem(domain,param,0) 
        
        # if sum is better, keep coefficients
        if objJ_temp < objJ:
            objJ = objJ_temp
            E_crit = E_crit_temp
            c_in = c_temp
            print("Better domain found with value: " + str(objJ))
            write_to_file(param.file_out_name_r,a[0,:],b[0,:],objJ,E_crit)
        
        param.n_steps = param.n_steps+1
        
        time_end = time.time()
        
        print("Elapsed time: " + str(time_end-time_start) + " seconds; " + str(objJ_temp))
        

# Gradient descent optimization routine
# Optimizes sums of eigenvalues by gradient descent method
def gradient_descent_optimization(c_in, domainClass, param):
    
    total_time_start = time.time()

    # set functions dependent on operator
    if param.operator == "DL":
        print("**********************************************")
        print("* Shape Optimization for Dirichlet Laplacian *")
        print("**********************************************")
        eigval_gradient = eigval_gradient_DirichletLaplacian
    elif param.operator == "MDL":
        print("*******************************************************")
        print("* Shape Optimization for Magnetic Dirichlet Laplacian *")
        print("*******************************************************")
        eigval_gradient = eigval_gradient_MagneticDirichletLaplacian

    # Domain initialization
    c = c_in
    shape_c = np.shape(c)

    # Initialize stepwise save variables
    c_iter = np.zeros((param.max_steps,shape_c[0],shape_c[1]))
    d_iter = np.zeros((param.max_steps,shape_c[0],shape_c[1]))
    sum_n_iter = np.zeros((param.max_steps,1))
    E_crit_iter = np.zeros((param.max_steps,param.n))
    mult_n_iter = np.zeros((param.max_steps,1))

    # Gradient descent loop
    n_steps = 0; beta_opt = 10**(-5); beta_mult = 1.0; beta_max = 1.0 	# some starting values
    phase = 'gradadd'
    while n_steps < param.max_steps:

        print("--- Iteration "+str(n_steps)+" ---")

        # plot current domain
        # domain = Domain(a,b,param)
        # domain.rescale_domain()               # rescale and redistribute collocation points
        # domain.set_t_col(t_init=None, do_node_repel=domain.param.do_node_repel, node_repel_n_max_iter=domain.param.node_repel_n_max_iter)               # redefine t_col points for equi.arc.dist
        # domain.set_domain_variables()         # set collocation and source points
        # plt.plot(domain.Gamma_col[0,:],domain.Gamma_col[1,:],'b')
        # plt.show()

        # solve direct problem
        print("Compute eigenvalues of current domain")
        time_start = time.time()

        domain = domainClass(c, param)
        domain.rescale_domain()               # rescale and redistribute collocation points 
        domain.set_t_col(t_init=None, do_node_repel=domain.param.do_node_repel, node_repel_n_max_iter=domain.param.node_repel_n_max_iter)               # redefine t_col points for equi.arc.dist
        domain.set_domain_variables()         # set collocation and source points        
        
        sum_n, E_crit, mult_n, V, dE_dB = direct_problem(domain, param, 1)
        
        sum_n_iter[n_steps,0] = sum_n
        E_crit_iter[n_steps,:] = E_crit[0:param.n]
        mult_n_iter[n_steps,0] = mult_n

        print("Objective function values: ")
        print(sum_n_iter[0:n_steps+1,0])

        write_to_file(param.file_out_name,c,sum_n,E_crit)

        # check if eigenvalue gap for n-th eigenvalue is small
        gapconverged = True
        if mult_n > 1:
            gapconverged = (abs(E_crit_iter[n_steps,-1]-E_crit_iter[n_steps,-mult_n]) < 100*param.eps)
        print("Current eigenvalue gap is: " + str(abs(E_crit_iter[n_steps,-1]-E_crit_iter[n_steps,-mult_n])) + ", gapconverged: " + str(gapconverged))
        
        # check on convergence
        convergence_condition = (n_steps > 2) and gapconverged and (abs(sum_n_iter[n_steps-1] - sum_n_iter[n_steps]) < param.eps)
        print("Converged: " + str(convergence_condition) )
        if convergence_condition: 
            print("Objective function has converged within eps-tolerance!")
            break

        time_end = time.time()
        print("Elapsed time: " + str(time_end-time_start) + " seconds")

        # do gradient descent
        print("Compute gradient and apply line search")
        time_start = time.time()
        n_V = max(mult_n,param.sum_length,param.n_V_current)            # param.sum_length = max(param.sum_length, mult_n); n_V = param.sum_length; 
        print("Multiplicity: " + str(mult_n) + " - Sum length: " + str(param.sum_length) + " - p0: " + str(param.p0)) 
        d_all = np.zeros((n_V,shape_c[0],shape_c[1]))  # individual eigenvalue gradients
        d = np.zeros((shape_c[0],shape_c[1]))        # gradient for objective

        # compute all individual gradients
        for m in range(n_V):
            d_all[m,...] = eigval_gradient(domain,param,E_crit[param.n-1-m],V[:,m], dE_dB[m])

        d_all_temp = np.zeros((n_V,shape_c[0]*shape_c[1]))
        for k in range(n_V):
            d_all_temp[k,:] = d_all[k,...].reshape((shape_c[0]*shape_c[1],))       # make gradients (matrix-form) to long vectors

        # compute mean norm      
        d_norms = np.zeros((n_V,))
        for m in range(n_V):    
            d_norms[m] = np.linalg.norm(d_all_temp[m,:])
        max_norm = np.max(d_norms)

        # assemble full gradient or optimize in case of eigenvalues that got very close
        beta_reduce = 1.0                           # reduces beta_opt in case multiplicity is detected, but grad_opt is not viable anymore
        if n_V == param.sum_length:
            print("no multiplicity detected - adding gradients")
            phase = 'gradadd'
            beta_reduce = 0.9
        elif (abs(sum_n_iter[n_steps-1] - sum_n_iter[n_steps]) < 20*param.eps):
            print("last decrease was < 20*param.eps - adding gradients")
            phase = 'gradadd'
            beta_reduce = 0.5
        else:
            print("multiplicity detected - optimizing gradient")
            phase = 'gradopt'
        
        # depending on the current phase, build the search direction
        if phase == 'gradadd':                       
            for m in range(param.sum_length):    
                d = d + (E_crit[param.n-1-m]/sum_n)**(param.p0-1) * param.sum_weight[max(-1-m,-param.sum_length)] * d_all[m,...]
            beta_mult = param.beta_mult_phase_gradadd
        elif phase == 'gradopt': 
            if param.sum_length == 1:
                param.n_V_current = n_V                                                      # always optimize 
            d, f_opt = optimize_d_direction_combination(d_all_temp.T)
            d = d.reshape((shape_c[0],shape_c[1]))                                          # transform long vector back to matrix-form
            d = np.abs(f_opt)*d 
            print("gradients norm comparison:")
            print(d_norms)
            print(np.abs(f_opt))  
            beta_mult = param.beta_mult_phase_gradopt
                        
            # use normal gradient decrease mult_tol if necessary
            #if np.abs(f_opt) < 0.05*np.min(d_norms):
            #    print("gradient optimization is not good enough - continue with added gradients, decrease mult_tol")
            #    d = np.zeros((shape_c[0],shape_c[1])) 
            #    for m in range(param.sum_length):    
            #        d = d + param.sum_weight[max(-1-m,-param.sum_length)]*d_all[m,...]
            #    beta_reduce = 0.5
                                     

        print(d)
    
        # optimize on line
        beta_max = beta_mult/np.linalg.norm(d)  #max_norm #  
        print("beta_max = " + str(beta_max)) 
        beta_interval = np.array([0, beta_max])    # beta interval in which line search is performed
        beta_opt = gr_min_line_search(c, domainClass, d, param, n_V, beta_interval); 
        print("beta_opt = " + str(beta_opt)) 
        
        c = c - beta_reduce*beta_opt*d                    # update coefficients
        
        # increase beta_max of current phase if beta_opt is was too close in last step
        #if beta_opt > 0.95 * beta_max:              
        #    if phase == 'gradadd':
        #        param.beta_mult_phase_gradadd = 2.0 * param.beta_mult_phase_gradadd # max_norm * beta_max # 
        #    elif phase == 'gradopt': 
        #        param.beta_mult_phase_gradopt = 2.0 * param.beta_mult_phase_gradopt # max_norm * beta_max # 
        #else:
        #    if phase == 'gradadd':
        #        param.beta_mult_phase_gradadd = 2.0 * max_norm * beta_opt
        #    elif phase == 'gradopt': 
        #        param.beta_mult_phase_gradopt = 2.0 * max_norm * beta_opt
        
        # rescale new coefficients
        domain = domainClass(c, param)
        domain.rescale_domain()               # rescale and redistribute collocation points 
        domain.set_t_col(t_init=None, do_node_repel=domain.param.do_node_repel, node_repel_n_max_iter=domain.param.node_repel_n_max_iter)               # redefine t_col points for equi.arc.dist
        domain.set_domain_variables()         # set collocation and source points
        c = domain.get_c()
        
        # save new coefficients and gradient
        c_iter[n_steps,...] = c
        d_iter[n_steps,...] = d
        
        param.n_steps = n_steps
        n_steps = n_steps+1
        
        time_end = time.time()
        print("Elapsed time: " + str(time_end-time_start) + " seconds")

    total_time_end = time.time()
    print("Total elapsed time: " + str(total_time_end-total_time_start) + " seconds")

    return mult_n


# Gradient descent line search
# search for minimum of objective function along negative gradient d
def gr_min_line_search(c, domainClass, d, param, mult_number, beta_interval):
    
    param_temp = param      #Parameters()
    #param_temp.set_sum_length(mult_number)

    gr = (np.sqrt(5) + 1) / 2
    # beta stores interval and midpoints on which golden ratio search works
    beta = np.array([beta_interval[0], beta_interval[1], beta_interval[1] - (beta_interval[1] - beta_interval[0]) / gr, beta_interval[0] + (beta_interval[1] - beta_interval[0]) / gr])
    sum_beta = np.array([0.0, 0.0, 0.0, 0.0]);    # stores intermediate objective function values 
    sum_beta_needed = [0,0,1,1]    # tells which midpoints in GR search have to be newly calculated

    # golden ratio search
    phase = 'presearch'
    while abs(beta[1] - beta[0]) > param_temp.beta_rel*beta[0] or beta[0] == 0.0 :
        # calculate new value for new betas       
        if sum_beta_needed[2] == 1:
            c2 = c - beta[2]*d
            domain2 = domainClass(c2, param)
            domain2.rescale_domain()               # rescale and redistribute collocation points 
            domain2.set_t_col()               # redefine t_col points for equi.arc.dist
            domain2.set_domain_variables()         # set collocation and source points
            sum_n2, _, _, _, _  = direct_problem(domain2,param_temp,0)
            sum_beta[2] = sum_n2

        if sum_beta_needed[3] == 1:
            c3 = c - beta[3]*d
            domain3 = domainClass(c3, param)
            domain3.rescale_domain()               # rescale and redistribute collocation points 
            domain3.set_t_col()               # redefine t_col points for equi.arc.dist
            domain3.set_domain_variables()         # set collocation and source points
            sum_n3, _, _, _, _ = direct_problem(domain3,param_temp,0)
            sum_beta[3] = sum_n3
        
        if phase == 'presearch':                # increase search interval size, if right boundary is lower than left
            if sum_beta[2] > sum_beta[3]:
                beta = 2.0*beta
                sum_beta_needed = [0,0,1,1];  
            else:
                phase = 'search'
                
        if phase == 'search':
            # reassign beta interval and midpoints
            if sum_beta[2] < sum_beta[3]:
                beta[1] = beta[3]
                beta[3] = beta[2]
                sum_beta[3] = sum_beta[2]
                beta[2] = beta[1] - (beta[1] - beta[0]) / gr
                sum_beta_needed = [0,0,1,0];  
            else:
                beta[0] = beta[2]
                beta[2] = beta[3]
                sum_beta[2] = sum_beta[3]
                beta[3] = beta[0] + (beta[1] - beta[0]) / gr;
                sum_beta_needed = [0,0,0,1];  
        
        print(beta)
        print(sum_beta[2:])

    beta_opt = (beta[1] + beta[0]) / 2
   
    return beta_opt

    
# Find eigenfrequencies for given domain 
# last argument can be used to give only n-th eigenfreq without koeff for eigenfunction
def direct_problem(domain,param,V_needed):

    time_start = time.time()

    # for DL and MDL matrix A computations
    gamma = np.hstack( (domain.Gamma_col, domain.Gamma_interior) )
    diff_matrix_x = np.subtract.outer( gamma[0,:], domain.p_source[0,:])
    diff_matrix_y = np.subtract.outer( gamma[1,:], domain.p_source[1,:])
    diff_xy2 = diff_matrix_x**2+diff_matrix_y**2
    diff_xy = np.sqrt(diff_xy2);
    cross_prod = np.multiply.outer(gamma[1,:], domain.p_source[0,:]) - np.multiply.outer(gamma[0,:], domain.p_source[1,:])
    f_12 = np.exp(0.5*1j*param.B0 * cross_prod)
    rho2 = 0.5*param.B0*diff_xy2

    # set functions dependent on operator
    if param.operator == "DL":
        A = lambda E: A_DirichletLaplacian(diff_xy, E)
        eigf_norm = eigf_norm_DirichletLaplacian
    elif param.operator == "MDL":
        A = lambda E: A_MagneticDirichletLaplacian(f_12, rho2, param.B0, E)
        eigf_norm = eigf_norm_MagneticDirichletLaplacian






        
    def sing1A(E, k_sing):      # only scalar E
        B = A(E)

        #if B.ndim==2:
        sing_kappa = np.zeros((k_sing,))
        Q, _ = np.linalg.qr(B)             
        _, s, _ = scipy.sparse.linalg.svds(Q[:domain.param.N_col,:], k=k_sing, which='SM', solver='arpack')
        sing_kappa = s
        
        #else:   
        #    n_kappa = np.shape(E)[0]
        #    sing_kappa = np.zeros((n_kappa,k_sing))
        #    
        #    print(np.shape(sing_kappa))
        #    for k in range(n_kappa):
        #        Q, _ = np.linalg.qr(B[k,...]) 
        #        _, s, _ = scipy.sparse.linalg.svds(Q[:domain.param.N_col,:], k=k_sing, which='SM')
        #        sing_kappa[k,:] = s

        return sing_kappa
        
    # find eigenvalues by searching sing1A for minima
    interval = [param.interval_start, param.interval_end];  # search interval for eig.frq
    tol = 10**(-12);                          # eigval accuracy tolerance
    filter_tol = 2*tol;                   # eigval filter tolerance
    
    # first eigenvalue search
    E_crit = search_eigval_interval(sing1A, interval, param.Nk, tol)

    # search iteratively on finer mesh if necessary
    dlambda = (interval[1]-interval[0])/param.Nk; 
    Nk = 30                                         # sing1A resolution for finer mesh
    n_Iter_fine =  0                                # counts refining iterations
    while len(E_crit)<param.n or n_Iter_fine < param.n_Iter_fine_min:   
        
        E_crit_temp=np.array([])        # where new mins are temporarily saved
        
        # search around already found minima 
        for m in range(min(len(E_crit),param.n)):
            
            if sing1A(E_crit[m], 2)[1] > 0.1:           # skip finer search around simple eigenvalues (where second singular value is large)
                continue
            
            interval = [max(E_crit[m]-5*dlambda, param.interval_start), min(E_crit[m]+5*dlambda,param.interval_end)]

            E_crit_temp = np.append(E_crit_temp,search_eigval_interval(sing1A, interval, Nk, tol))

            # filter out already found E_crit and ones possibly found below a proven absolute boundary
            #dist_to_E_crit = np.min(np.abs(np.subtract.outer(E_crit_temp,E_crit)),1)            # minimal distance to another E_crit in already known E_crit
            #E_crit_temp=E_crit_temp[np.logical_and(np.greater(dist_to_E_crit,filter_tol) , np.greater(E_crit_temp, 4.2))];
            
            # add newly found E_crit_temp to E_crit   
            E_crit = np.append(E_crit, E_crit_temp)
            
            # filter out all E_crit that are the same within filter_tol and those which have sing1A too bad 
            E_crit2 = [0.0]
            for k in range(len(E_crit)):
                if np.min(np.abs(np.subtract.outer(E_crit[k],E_crit2))) >= filter_tol: 
                    E_crit2 = np.append(E_crit2, E_crit[k])
            E_crit = np.sort(E_crit2[1:])

            # quit if enough new E_crit were found
            if len(E_crit)>=param.n:
                break
        
        print("dlambda = " + str(dlambda)); 
        dlambda = 0.35*dlambda             # new sing1A resolution in lambda space
        n_Iter_fine = n_Iter_fine + 1
        
        # leave loop, if refining does not find more eigenvalues and print an error
        if dlambda < filter_tol:
            print("ERROR: could not find enough E_crit, dlambda < filter_tol !!!")
            print("dlambda = " + str(dlambda) + " < " + str(filter_tol) + " = filter_tol")
            print("E_crit = ")
            print(E_crit)
            break

    time_end = time.time()
    if param.verbose == 1:
        print("Elapsed time: " + str(time_end-time_start) + " seconds")
        print(E_crit)
    
    # if not enough eigenfrequencies found, fill up E_crit with param.interval_end
    if len(E_crit)<param.n:
        E_crit = np.append(E_crit, (param.n-len(E_crit))*[param.interval_end])
        print("Using upper interval boundary for remaining eigenfrequencies")
 
    

    # save multiplicity of n-th eigenvalue and all eig.frequencies in mult
    mult_n = 1
    k = param.n-1
    while k > 0 and np.abs(E_crit[k]-E_crit[k-1]) < param.mult_tol:
        mult_n = mult_n + 1
        k = k - 1
    
    # compute coefficients for L2-normalized eigenfunctions
    bnd = 0.9
    dh = 0.01
    n_V = max(mult_n,param.sum_length,param.n_V_current)            #param.sum_length = max(param.sum_length, mult_n); n_V = param.sum_length;
    V = np.zeros((domain.N_p_source,n_V), dtype=complex)
    dE_dB = np.zeros((n_V,))
            
    if V_needed == 1:
        for m in range(n_V):
            s = np.nan
            E_temp = E_crit[param.n-1-m]

            # try to catch NaN singular values by wiggling the EV a little
            while np.isnan(s) or s == 0.0:
                B = A(E_temp)
                Q, R = np.linalg.qr(B)
                _, s, w_temp = scipy.sparse.linalg.svds(Q[:domain.param.N_col,:], k=1, which='SM')
                print("minimal singular value: " + str(s) + ", E:" + str(E_temp))
                E_crit[param.n-1-m] = E_temp
                E_temp = E_temp + 10**(-7)*rand()
            
            V_temp = np.linalg.solve(R, w_temp.conj().T)
            
            L2norm, dE_dB_temp = eigf_norm(E_crit[param.n-1-m],V_temp,domain,dh,bnd)
            print("L2 norm before: " + str(L2norm) )
            V[:,m] = V_temp.T / L2norm
            dE_dB[m] = dE_dB_temp
            
            #dn_eigf(domain.Gamma_col[0,:],domain.Gamma_col[1,:],E_crit[param.n-1-m],V[:,m],domain)
    
    sum_n = np.sum( (param.sum_weight[(param.n-param.sum_length):param.n]*E_crit[(param.n-param.sum_length):param.n]**param.p0) )**(1.0/param.p0) 
    #print(sum_n);print(E_crit)
    return sum_n, E_crit, mult_n, V, dE_dB

# W stencil like search with bisection and recursive splitting if W value pattern is found
# splitting doesn't appear to work always

def Wsearch_eigval_interval(f, interval, interval_values, tol):
    
    E_crit = np.array([])
    
    if len(interval) < 2:
        print("interval broken ?!")
        return E_crit
    
    
    if interval[1]-interval[0] < tol:
        E_crit = 0.5*(interval[1]+interval[0])
    else:
        E_sample = np.linspace(interval[0], interval[1], 5)        
        a=np.zeros((len(E_sample), ))
        a[0] = interval_values[0]; a[-1] = interval_values[-1];     # copy old values for outer stencil points
        for j in range(1,len(E_sample)-1):
            a[j] = f(E_sample[j])
        E_local_min = argrelmax(-a)
    
        print(E_sample); print(a); print(E_local_min)
    
        for k in E_local_min:
            if k==0 or k==4:
                interval_new        = np.hstack((E_sample[k/2], E_sample[k/2+2]))
                interval_values_new = np.hstack((a[k/2], a[k/2+2]))
            else:
                interval_new        = np.hstack((E_sample[k-1], E_sample[k+1]))
                interval_values_new = np.hstack((a[k-1], a[k+1]))
               
            E_crit = np.hstack( (E_crit, Wsearch_eigval_interval(f, interval_new, interval_values_new, tol)) )
            
    return E_crit


# Search for eigenvalues in interval
# searches for minima of f = log(|det(A(Energies))|) in interval
def search_eigval_interval(sing1A, interval, Nk, tol):

    dlambda_cur = (interval[1] - interval[0]) / Nk

    if interval[0] >= 50.0:              # at high energy, finer evaluation at lower boundary
        Energies_start = interval[0] - 1 + np.exp(np.linspace(0, np.log(1+6*dlambda_cur), 50))      # finer eval at 
        Energies_end   = np.linspace(interval[0]+7*dlambda_cur, interval[1], Nk+1)
        Energies = np.append(Energies_start, Energies_end)
    else: 
        Energies = np.linspace(interval[0], interval[1], Nk+1)
    
    a=np.zeros((1,len(Energies)))
    
    # evaluate function on interval in Nk equidistant points
    #a = f(Energies)
    for j in range(len(Energies)):
        a[:,j] = sing1A(Energies[j],1)

    a[np.isnan(a)] = 10**(-8)       # overwrite, if we get nan at center
        
    if debug > 0:
        #plot f = log(|det(A)|) over Energies
        plt.plot(Energies,a[0,:])
        #plt.plot(Energies,a[1,:])
        plt.show()
    
    # find local min of function
    E_local_max = Energies[argrelmax(a[0,:])]
    E_local_min = Energies[argrelmax(-a[0,:])]

    if len(E_local_max) > 0:
        if E_local_max[0] > E_local_min[0]:
            E_local_max = np.append(interval[0], E_local_max)
        if E_local_max[-1] < E_local_min[-1]:
            E_local_max = np.append(E_local_max, interval[1])
    else:
        E_local_max = interval
    
    
    mins = np.array([]);

    # search only if there is more than one local min between interval boundaries found
    if len(E_local_min) >= 1:
        # search for local minima between subsequent local maxima and also last
        # maximum and right interval boundary
        
        for j in range(len(E_local_min)):
            local_search_interval        = [max(E_local_max[j],E_local_min[j]-dlambda_cur), min(E_local_max[j+1],E_local_min[j]+dlambda_cur)]

            #print(local_search_interval)
            local_min = gr_min_search(lambda E: sing1A(E,1), local_search_interval, tol)
            if sing1A(local_min,1) < 10**(-4):
                mins = np.append(mins,local_min);
            else:
                print("bad E_crit found! singA 10**(-4) tolerance")
            #local_search_interval_values = [sing1A(local_search_interval[0],1)[0], sing1A(local_search_interval[1],1)[0]]
            #mins = np.append(mins,Wsearch_eigval_interval(lambda E: sing1A(E,1), local_search_interval, local_search_interval_values, tol))
            #res = scipy.optimize.minimize(f, 0.5*(local_search_interval[0]+local_search_interval[1]), method = 'BFGS')
            
            #res  = scipy.optimize.minimize_scalar(lambda E: sing1A(E,1), (local_search_interval[0], local_search_interval[1]))
            #mins = np.append(mins,res.x)
                
        #print(mins)

        # delete errenous mins at interval boundaries
        mins = mins[ np.logical_and(np.greater(np.subtract(mins,interval[0]),2*tol) , np.greater(np.subtract(interval[1],mins),2*tol)) ]  

    return mins





#######################################
###   Routines for eigenfunctions   ###
#######################################

##################################################################
### evaluate a MFS eigenfunction of DirichletLaplacian ###



# Gradient of eigenvalues wrt Fourier coefficients
# computes the gradient of eigenvalue lambda = k_crit^2 wrt Fourier coefficients
def eigval_gradient_DirichletLaplacian(domain,param,E_crit, V,*args):

    eigval = E_crit
    Gamma_int = domain.gamma_xy(param.t_int)

    Normal_temp = np.vstack((domain.dt_gamma_y(param.t_int),-domain.dt_gamma_x(param.t_int)))
    Normal_norm = np.sqrt(Normal_temp[0,:]**2+ Normal_temp[1,:]**2);
    Normal_temp[0,:] = Normal_temp[0,:]/Normal_norm
    Normal_temp[1,:] = Normal_temp[1,:]/Normal_norm
    Normal_int = Normal_temp

    dn_eigf_temp = lambda X,Y: dn_eigf_DirichletLaplacian(X,Y,E_crit, V,domain,Normal_int);  
    line_weight = np.sqrt( np.sum(domain.dt_gamma_xy(param.t_int)**2, axis = 0) )    # norm of boundary tangent vector (integral weight that accounts for curvature)
    
    # evaluate formula for derivatives with quadrature
    # constant part of integrand
    A_1 = (eigval - np.abs(dn_eigf_temp(Gamma_int[0,:],Gamma_int[1,:])**2)) * line_weight

    if isinstance(domain, GeneralizedDomain):
        d = np.zeros((4,param.len_ab))
        A_2x = A_1 * Normal_int[0,:]
        A_2y = A_1 * Normal_int[1,:]

        ### x coefficients ###
        # upper half with cos  
        for j in range(param.len_ab):
            d[0,j] = param.dt_int * np.sum( A_2x  *  np.cos(j*param.t_int) , 1 )
        
        # lower half with sin
        for j in range(param.len_ab):
            d[1,j] = param.dt_int * np.sum( A_2x  *  np.sin(j*param.t_int) , 1 ) 

        ### y coefficients ###
        # upper half with cos  
        for j in range(param.len_ab):
            d[2,j] = param.dt_int * np.sum( A_2y  *  np.cos(j*param.t_int) , 1 )
        
        # lower half with sin
        for j in range(param.len_ab):
            d[3,j] = param.dt_int * np.sum( A_2y  *  np.sin(j*param.t_int) , 1 ) 

    else:
        d = np.zeros((2,param.len_ab))
        A_2 = A_1 * np.sum(Gamma_int*Normal_int,0)/domain.r(param.t_int) 

        # upper half with cos  
        for j in range(param.len_ab):
            d[0,j] = param.dt_int * np.sum( A_2  *  np.cos(j*param.t_int) , 1 )
        
        # lower half with sin
        for j in range(param.len_ab):
            d[1,j] = param.dt_int * np.sum( A_2  *  np.sin(j*param.t_int) , 1 )  

    return d




def eigf_DirichletLaplacian(X,Y,E,alpha, domain):

    n_grid = np.shape(X)
    Z = np.zeros(n_grid, dtype=complex)
    
    ind_comp = domain.outside_domain(X, Y)
        
    for k in range(n_grid[0]):
        diff_matrix_x = np.subtract.outer( X[k,:], domain.p_source[0,:])
        diff_matrix_y = np.subtract.outer( Y[k,:], domain.p_source[1,:])
        diff_xy = np.sqrt(diff_matrix_x**2+diff_matrix_y**2)
                
        Z[k,:] = np.dot(hankel1(0,np.sqrt(E)*diff_xy), alpha).T

    Z[ind_comp]=0.0           # set eigenfct. to zero outside of domain.
    return Z, ind_comp



# evaluate a MFS eigenfunction
# on domain boundary for testing purposes
def eigf_bnd_DirichletLaplacian(t,E,alpha,domain):

    bnd_xy = domain.gamma_xy(t)
    X = bnd_xy[0,:]
    Y = bnd_xy[1,:]

    diff_matrix_x = np.subtract.outer( X, domain.p_source[0,:])
    diff_matrix_y = np.subtract.outer( Y, domain.p_source[1,:])    
    diff_xy = np.sqrt(diff_matrix_x**2+diff_matrix_y**2)

    Z = np.dot(hankel1(0,np.sqrt(E)*diff_xy), alpha)

    return Z



# L^2-norm of a MFS eigenfunction
# summing over a discrete grid

def eigf_norm_DirichletLaplacian(E,alpha,domain,dh,bnd): #(E,alpha,p_source,dh,radius):

    x=np.arange(-bnd,bnd,dh)
    y=np.arange(-bnd,bnd,dh)

    X,Y = np.meshgrid(x,y)
    Z, ind_comp = eigf_DirichletLaplacian(X,Y,E,alpha, domain)
    dE_dB = 0.0

    # plot eigenfunction abs squared
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot_surface(X, Y, np.abs(Z**2), cmap='plasma', edgecolor='none')
    #ax.set_title('Eigenfunction')
    #plt.show()

    L2norm = dh * np.linalg.norm(Z,ord='fro')           # dh without square because     dh * sqrt(sum... ) = sqrt(sum dh**2... ) =~ sqrt(int ...)

    return L2norm, dE_dB


# normal derivative of a MFS eigenfunction on boundary
# vectorized
def dn_eigf_DirichletLaplacian(x,y,E,alpha,domain,normals):

    # compute negative gradient
    gradient_u = np.zeros((2,len(x)), dtype=complex)
    diff_matrix_x = np.subtract.outer( x, domain.p_source[0,:])
    diff_matrix_y = np.subtract.outer( y, domain.p_source[1,:])
    dist_temp = np.sqrt(diff_matrix_x**2+diff_matrix_y**2)
    gradient_u[0,:] = np.dot( np.subtract.outer( x, domain.p_source[0,:]) * np.sqrt(E) * hankel1(1,np.sqrt(E)* dist_temp)/ dist_temp , alpha) 
    gradient_u[1,:] = np.dot( np.subtract.outer( y, domain.p_source[1,:]) * np.sqrt(E) * hankel1(1,np.sqrt(E)* dist_temp)/ dist_temp , alpha) 

    # scalar product of gradients with normals
    z = - np.sum(gradient_u * normals, 0)
    
    return z




##################################################################
### evaluate a MFS eigenfunction of MagneticDirichletLaplacian ###

# Gradient of eigenvalues wrt Fourier coefficients
# computes the gradient of eigenvalue lambda = E_crit wrt Fourier coefficients
def eigval_gradient_MagneticDirichletLaplacian(domain,param,E_crit, V, dE_dB):    
        
    eigval = E_crit
    Gamma_int = domain.gamma_xy(param.t_int)

    Normal_temp = np.vstack((domain.dt_gamma_y(param.t_int),-domain.dt_gamma_x(param.t_int)))
    Normal_norm = np.sqrt(Normal_temp[0,:]**2+ Normal_temp[1,:]**2);
    Normal_temp[0,:] = Normal_temp[0,:]/Normal_norm
    Normal_temp[1,:] = Normal_temp[1,:]/Normal_norm
    Normal_int = Normal_temp

    dn_eigf_temp = lambda X,Y: dn_eigf_MagneticDirichletLaplacian(X,Y,E_crit, V,domain,Normal_int); 
    line_weight = np.sqrt( np.sum(domain.dt_gamma_xy(param.t_int)**2, axis = 0) )    # norm of boundary tangent vector (integral weight that accounts for curvature)
    
    # evaluate formula for derivatives with quadrature
    # constant part of integrand
    A_1 = (eigval - param.B0*dE_dB - np.abs(dn_eigf_temp(Gamma_int[0,:],Gamma_int[1,:])**2))  * line_weight 

    if isinstance(domain, GeneralizedDomain):
        d = np.zeros((4,param.len_ab))
        A_2x = A_1 * Normal_int[0,:]
        A_2y = A_1 * Normal_int[1,:]

        ### x coefficients ###
        # upper half with cos  
        for j in range(param.len_ab):
            d[0,j] = param.dt_int * np.sum( A_2x  *  np.cos(j*param.t_int) , 1 )
        
        # lower half with sin
        for j in range(param.len_ab):
            d[1,j] = param.dt_int * np.sum( A_2x  *  np.sin(j*param.t_int) , 1 ) 

        ### y coefficients ###
        # upper half with cos  
        for j in range(param.len_ab):
            d[2,j] = param.dt_int * np.sum( A_2y  *  np.cos(j*param.t_int) , 1 )
        
        # lower half with sin
        for j in range(param.len_ab):
            d[3,j] = param.dt_int * np.sum( A_2y  *  np.sin(j*param.t_int) , 1 ) 

    else:
        d = np.zeros((2,param.len_ab))
        A_2 = A_1 * np.sum(Gamma_int*Normal_int,0)/domain.r(param.t_int) 

        # upper half with cos  
        for j in range(param.len_ab):
            d[0,j] = param.dt_int * np.sum( A_2  *  np.cos(j*param.t_int) , 1 )
        
        # lower half with sin
        for j in range(param.len_ab):
            d[1,j] = param.dt_int * np.sum( A_2  *  np.sin(j*param.t_int) , 1 )  

    return d

# evaluate MFS eigenfunction on XY grid
def eigf_MagneticDirichletLaplacian(X,Y,E,alpha, domain):  

    n_grid = np.shape(X)
    Z = np.zeros(n_grid, dtype=complex)
    
    ind_comp = domain.outside_domain(X, Y)
    
    #out = np.zeros(np.shape(ind_comp))
    #out[ind_comp] = 1.0
    #print(np.shape(ind_comp))
    kappa = 0.5*E/domain.param.B0
    
    for k in range(n_grid[0]):
        diff_matrix_x = np.subtract.outer( X[k,:], domain.p_source[0,:])
        diff_matrix_y = np.subtract.outer( Y[k,:], domain.p_source[1,:])
        cross_prod = np.multiply.outer(Y[k,:], domain.p_source[0,:]) - np.multiply.outer(X[k,:], domain.p_source[1,:])
        f_12 = np.exp(0.5*1j*domain.param.B0 * cross_prod)
        
        rho2 = 0.5 * domain.param.B0*(diff_matrix_x**2+diff_matrix_y**2)
                
        Z[k,:] = np.dot( f_12*np.exp(-0.5*rho2)*hyperg_U(0.5-kappa,1,rho2)  , alpha).T

    Z[ind_comp]=0.0           # set eigenfct. to zero outside of domain.
    return Z, ind_comp


# evaluate MFS eigenfunction on domain boundary
def eigf_bnd_MagneticDirichletLaplacian(t,E,alpha, domain):  

    bnd_xy = domain.gamma_xy(t)
    X = bnd_xy[0,:]
    Y = bnd_xy[1,:]
    
    kappa = 0.5*E/domain.param.B0    

    diff_matrix_x = np.subtract.outer( X, domain.p_source[0,:])
    diff_matrix_y = np.subtract.outer( Y, domain.p_source[1,:])
    cross_prod = np.multiply.outer(Y, domain.p_source[0,:]) - np.multiply.outer(X, domain.p_source[1,:])
    f_12 = np.exp(0.5*1j*domain.param.B0 * cross_prod)
    
    rho2 = 0.5 * domain.param.B0*(diff_matrix_x**2+diff_matrix_y**2)
            
    Z = np.dot( f_12*np.exp(-0.5*rho2)*hyperg_U(0.5-kappa,1,rho2)  , alpha).T
        
    return Z

# L^2-norm of a MFS eigenfunction
# summing over a discrete grid
def eigf_norm_MagneticDirichletLaplacian(E,alpha,domain,dh,bnd):
    
    x=np.arange(-bnd,bnd,dh)
    y=np.arange(-bnd,bnd,dh)

    X,Y = np.meshgrid(x,y)
    Z, ind_comp = eigf_MagneticDirichletLaplacian(X,Y,E,alpha,domain)
    G = grad_eigf2_MagneticDirichletLaplacian(X,Y,E,alpha,domain)
    G[ind_comp] = 0.0           # set gradient zero outside domain   
    L2norm = dh * np.linalg.norm(Z,ord='fro')           # dh without square because     dh * sqrt(sum... ) = sqrt(sum dh**2... ) =~ sqrt(int ...)
    Z = Z / L2norm
    G = G / L2norm**2
    dE_dB = (E + 0.25*domain.param.B0**2 * dh**2 * np.sum((X**2+Y**2)*np.abs(Z)**2) - dh**2 * np.sum(G) ) /domain.param.B0 
    print("dE_dB = " + str(dE_dB)) 

            
    # plot eigenfunction abs squared
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot_surface(X, Y, np.abs(Z**2), cmap='plasma', edgecolor='none')
    #ax.set_title("Eigenfunction to E = " + str(E))
    #plt.show()
    # plot colored complex function plot
    #plot_complex_eigf(X,Y,Z)

    return L2norm, dE_dB


# normal derivative of a MFS eigenfunction on boundary
# vectorized
def dn_eigf_MagneticDirichletLaplacian(x,y,E,alpha,domain,normals): 

    # compute negative gradient
    gradient_u = np.zeros((2,len(x)), dtype=complex)
    diff_matrix_x = np.subtract.outer( x, domain.p_source[0,:])
    diff_matrix_y = np.subtract.outer( y, domain.p_source[1,:])
    
    cross_prod = np.multiply.outer(y, domain.p_source[0,:]) - np.multiply.outer(x, domain.p_source[1,:])
    f_12 = np.exp(0.5*1j*domain.param.B0 * cross_prod)
        
    rho2 = 0.5 * domain.param.B0 * (diff_matrix_x**2+diff_matrix_y**2)
    
    kappa = 0.5*E/domain.param.B0
    prefactor_term = f_12*np.exp(-0.5*rho2)
    #prefactor_term = np.subtract.outer(np.ones(np.shape(x)), prefactor_term)
    U_term_1_2 = hyperg_U(0.5-kappa,1,rho2)
    U_term_3_2 = hyperg_U(1.5-kappa,2,rho2)

    M_0 = 0.5 * domain.param.B0 * ( 1j*np.subtract.outer( np.zeros(np.shape(x)), domain.p_source[1,:]) -  np.subtract.outer( x, domain.p_source[0,:]) ) * U_term_1_2 - (0.5-kappa) * domain.param.B0 * np.subtract.outer( x, domain.p_source[0,:]) * U_term_3_2 
    M_1 = 0.5 * domain.param.B0 * ( 1j*np.add.outer( np.zeros(np.shape(y)), domain.p_source[0,:]) -  np.subtract.outer( y, domain.p_source[1,:]) ) * U_term_1_2 - (0.5-kappa) * domain.param.B0 * np.subtract.outer( y, domain.p_source[1,:]) * U_term_3_2 

    gradient_u[0,:] = np.dot(prefactor_term * M_0, alpha) 
    gradient_u[1,:] = np.dot(prefactor_term * M_1, alpha)
    
    # scalar product of gradients with normals
    z = np.sum(gradient_u * normals, 0)
    
    # check of analytic expression with finite difference dn_eigf
    #domain.Normal_col
    #dh = 0.01
    #x2 = x-dh*domain.Normal_col[0,:]
    #y2 = y-dh*domain.Normal_col[1,:]
    #x1 = x[np.newaxis].T
    #y1 = y[np.newaxis].T
    #x2 = x2[np.newaxis].T
    #y2 = y2[np.newaxis].T
    #
    #FD_dn_eigf = ( eigf(x1,y1,E,alpha, domain) - eigf(x2,y2,E,alpha, domain) ) / dh#
    #
    #print("normalized dn ableitungen")
    #print(z[0:10].T)
    #print(FD_dn_eigf[0:10,0])
    #print(np.abs(z-FD_dn_eigf[:,0]))
    #print("max FD error")
    #print(np.max(np.abs(z-FD_dn_eigf[:,0])))
    
    return z

# compute |grad u|^2 on domain
def grad_eigf2_MagneticDirichletLaplacian(X,Y,E,alpha,domain): 

    Gradient_U2 = np.zeros(np.shape(X))

    # compute gradient squared linewise
    for k in range(np.shape(X)[0]):
        
        gradient_u = np.zeros((2,len(X[k,:])), dtype=complex)
        diff_matrix_x = np.subtract.outer( X[k,:], domain.p_source[0,:])
        diff_matrix_y = np.subtract.outer( Y[k,:], domain.p_source[1,:])
    
        cross_prod = np.multiply.outer(Y[k,:], domain.p_source[0,:]) - np.multiply.outer(X[k,:], domain.p_source[1,:])
        f_12 = np.exp(0.5*1j*domain.param.B0 * cross_prod)
        
        rho2 = 0.5 * domain.param.B0 * (diff_matrix_x**2+diff_matrix_y**2)
    
        kappa = 0.5*E/domain.param.B0
        prefactor_term = f_12*np.exp(-0.5*rho2)
        U_term_1_2 = hyperg_U(0.5-kappa,1,rho2)
        U_term_3_2 = hyperg_U(1.5-kappa,2,rho2)

        M_0 = 0.5 * domain.param.B0 * ( 1j*np.subtract.outer( np.zeros(np.shape(X[k,:])), domain.p_source[1,:]) -  np.subtract.outer( X[k,:], domain.p_source[0,:]) ) * U_term_1_2 - (0.5-kappa) * domain.param.B0 * np.subtract.outer( X[k,:], domain.p_source[0,:]) * U_term_3_2 
        M_1 = 0.5 * domain.param.B0 * ( 1j*np.add.outer( np.zeros(np.shape(Y[k,:])), domain.p_source[0,:]) -  np.subtract.outer( Y[k,:], domain.p_source[1,:]) ) * U_term_1_2 - (0.5-kappa) * domain.param.B0 * np.subtract.outer( Y[k,:], domain.p_source[1,:]) * U_term_3_2 

        gradient_u[0,:] = np.dot(prefactor_term * M_0, alpha)[:,0] 
        gradient_u[1,:] = np.dot(prefactor_term * M_1, alpha)[:,0] 
    
        # take norm of gradients 
        Gradient_U2[k,:] = np.sum(np.abs(gradient_u)**2, 0)
   
    return Gradient_U2




# Build matrix function of E (imposing boundary conditions), only scalar E
def A_DirichletLaplacian(diff_xy, E):
    A_matrix = hankel1(0,np.sqrt(E) * diff_xy);
    return A_matrix

def A_MagneticDirichletLaplacian(f_12, rho2, B0, E):              
    A_matrix = np.zeros(np.shape(rho2), dtype = 'complex')        
    kappa = 0.5*E/B0
    A_matrix = f_12*np.exp(-0.5*rho2)*hyperg_U(0.5-kappa,1,rho2) 
    return A_matrix






















#   Golden ratio search for minimum in interval 
#   up to tolerance
def gr_min_search(f, interval, tol):
    gr = (np.sqrt(5) + 1) / 2;
    a = interval[0];
    b = interval[1];
    
    c = b - (b - a) / gr;
    d = a + (b - a) / gr;
    f_c = f(c)
    f_d = f(d)
    while np.abs(b - a) > tol:
        if f_c < f_d:
            b = d;
            d = c;
            c = b - (b - a) / gr;
            f_d = f_c
            f_c = f(c)
        else:
            a = c;
            c = d;
            d = a + (b - a) / gr; 
            f_c = f_d
            f_d = f(d)

    #minimum = (b + a) / 2;
    f_a = f(a); f_b = f(b);
    minimum = (b*f_a + a*f_b)/(f_a + f_b);      # assuming abs(x) like minimum -> linear interpolation to find approximate root
    
    return minimum;


#   Golden ratio search for maximum in interval 
#   up to tolerance
def gr_max_search(f, interval, tol):
    gr = (np.sqrt(5) + 1) / 2;
    a = interval[0];
    b = interval[1];
    
    c = b - (b - a) / gr;
    d = a + (b - a) / gr;
    f_c = f(c)
    f_d = f(d)
    while np.abs(b - a) > tol:
        if f_c > f_d:
            b = d;
        else:
            a = c;

        c = b - (b - a) / gr;
        d = a + (b - a) / gr;

    maximum = (b + a) / 2;

    return maximum;








###########################################################################
##    Read/Write functions

def write_to_file(filename,c,sum_n,E_crit):
    file_out_ab = filename + "_ab.txt"
    file_out_val = filename + "_val.txt"

    shape_c = np.shape(c)
    
    # write c coefficients to file (rows are a,b's )
    f_ab = open(file_out_ab,'a')
    s = ""
    for k in range(shape_c[0]-1):
        for i in range(shape_c[1]-1):
            s = s+str(c[k,i])+", "
        s=s+str(c[k,-1])+"; "
    for i in range(shape_c[1]-1):
        s = s+str(c[-1,i])+", "
    s=s+str(c[-1,-1])

    f_ab.write(s+'\n')
    f_ab.close()

    # write objective function values to file
    f_val = open(file_out_val,'a')
    s = str(sum_n)+"; "
    for i in range(len(E_crit)-1):
        s = s+str(E_crit[i])+", "
    if len(E_crit)>0:
        s=s+str(E_crit[-1])
    f_val.write(s+'\n')
    f_val.close()


def read_from_file(filename, line_number):
    f_c = open(filename,'r')
    alllines = f_c.readlines()
    
    c_strings = alllines[line_number].strip().split(';')
    n_c = len(c_strings)                        # number of coefficient vectors
    len_c = len(c_strings[0].split(','))        # length of coefficient vectors
    c = np.zeros((n_c, len_c),dtype=float)
    for k in range(n_c):                        # read coefficient vectors
        c_strings_list = c_strings[k].split(',')
        for i in range(len(c_strings_list)):
            c[k,i] = float(c_strings_list[i])

    f_c.close()

    return c

# read values and ignore ; and ,
def read_from_file_val(filename, line_number):
    f_c = open(filename,'r')
    alllines = f_c.readlines()
    
    numbers_string = ""
    c_strings = alllines[line_number].strip().split(';')
    
    objval = float(c_strings[0].strip())    
    numbers_string_list = c_strings[1].strip().split(',')
    a = np.zeros((len(numbers_string_list),),dtype=float)
    
    for k in range(len(numbers_string_list)):                        # read coefficient vectors
        a[k] = float(numbers_string_list[k])

    f_c.close()

    return objval , a

###########################################################################


###########################################################################
##    Special functions

##
## Whittaker W function
## not used anymore, directly with Kummer U implemented
##
#def whitw_E0(E,x):
#    shape_x = np.shape(x)
#    whitw_matrix = np.zeros(shape_x)
#    
#    ### with GSL Kummer U function (vectorized)
#    #time_start_indiv = time.time()
#    whitw_matrix = np.exp(-0.5*x)*np.sqrt(x)*hyperg_U(0.5-E,1,x)      
#    #time_end_indiv = time.time()
#    #print("Elapsed time: " + str(time_end_indiv-time_start_indiv) + " seconds")
#    
#    ### with SCIPY Kummer U function ###
#    
#    #time_start_indiv = time.time()
#    #for i in range(shape_x[0]):
#    #    for j in range(shape_x[1]):
#    #        whitw_matrix[i,j] = np.exp(-0.5*x[i,j])*np.sqrt(x[i,j])*hyperu(0.5-E,1,x[i,j])      
#    #time_end_indiv = time.time()
#    #print("Elapsed time: " + str(time_end_indiv-time_start_indiv) + " seconds")
#    
#    ### with MPMATH Kummer U function (100x slower) ###
# 
#    #time_start_indiv = time.time()
#    #hyperu_vec = np.frompyfunc(mpmath.hyperu, 3, 1)
#    #whitw_matrix = np.exp(-0.5*x)*np.sqrt(x)*hyperu_vec(0.5-E,1,x).astype(np.float)
#    #time_end_indiv = time.time()
#    #print("Elapsed time: " + str(time_end_indiv-time_start_indiv) + " seconds")
#    #print(np.max(np.abs(whitw_matrix-whitw_matrix2)))
#       
#    ### with GSL Kummer U function ###
#    
#    #time_start_indiv = time.time()
#    #for i in range(shape_x[0]):
#    #    for j in range(shape_x[1]):
#    #        whitw_matrix[i,j] = np.exp(-0.5*x[i,j])*np.sqrt(x[i,j])*hyperg_U(0.5-E,1,x[i,j])      
#    #time_end_indiv = time.time()
#    #print("Elapsed time: " + str(time_end_indiv-time_start_indiv) + " seconds")
#    
#    #print(whitw_matrix)
#    
#    return whitw_matrix
    
def optimize_d_direction(d):
    def f(x):
        return -np.min(np.dot(d.T,x))/np.sqrt(np.dot(x.T,x))        # -min, so that points in gradients' direction
    
    m = np.shape(d)[0]
    d_opt = np.zeros((1,m))
    f_opt = 1000.0

    for k in range(100):
        x0 = np.random.rand(1,m)
        res = scipy.optimize.minimize(f, x0, tol=1e-6)
        d_opt_temp = np.array(res.x)[np.newaxis]
        d_opt_temp = d_opt_temp/np.linalg.norm(d_opt_temp)
        f_opt_temp = f(d_opt_temp.T)
        #print(f_opt_temp)
        if f_opt_temp < f_opt:
            f_opt = f_opt_temp
            d_opt = d_opt_temp
            
    print("final alignment value: ")
    print(f_opt)
    
    return d_opt



## optimize direction within linear combinations of gradients
def optimize_d_direction_combination(d):
    dd_mat = np.dot(d.T,d)
    
    def f(alpha):
        return -np.min(np.dot(dd_mat,alpha))/np.linalg.norm(np.dot(d,alpha))        # -min, so that points in gradients' direction
    
    m = np.shape(d)[0]
    n_d = np.shape(d)[1]
    d_opt = np.zeros((1,m))
    f_opt = 1000.0

    for k in range(100):
        alpha0 = np.random.rand(1,n_d)
        res = scipy.optimize.minimize(f, alpha0, tol=1e-8)
        alpha_opt_temp = np.array(res.x)
        d_opt_temp     = np.dot(d,alpha_opt_temp)
        d_opt_temp     = d_opt_temp/np.linalg.norm(d_opt_temp)
        f_opt_temp     = f(alpha_opt_temp)
        #print(f_opt_temp)
        if f_opt_temp < f_opt:
            f_opt = f_opt_temp
            d_opt = d_opt_temp
            alpha_opt = alpha_opt_temp
            
    #print("final alignment value: ")
    #print(f_opt)
    
    return d_opt, f_opt



##############################################################################
############ Functions for colorful complex function plots. ##################
##############################################################################




def Hcomplex(z):# computes the hue corresponding to the complex number z
    H = np.angle(z) / (2*np.pi) + 1
    return np.mod(H, 1)


def domaincol_classic(w, s):#Classical domain coloring
    # w is the  array of values f(z)
    # s is the constant saturation
    
    def g(x):
        return (1- 1/(1+x**2))**0.2
    
    H = Hcomplex(w)
    S = s * np.ones(H.shape)
    V = g(np.absolute(w)**2)
    modul = np.absolute(w)
    V = (1.0-1.0/(1+modul**2))**0.2
    # the points mapped to infinity are colored with white; hsv_to_rgb(0, 0, 1)=(1, 1, 1)=white

    HSV = np.dstack((H, S, V))
    RGB = hsv_to_rgb(HSV)
    return RGB

def plot_complex_eigf(X,Y,Z,s=0.9):


    def matrixflip(m,d):
        myl = np.array(m)
        if d=='v': 
            return np.flip(myl, axis=0)
        elif d=='h':
            return np.flip(myl, axis=1)

    domc = domaincol_classic(Z, s)
    domc = matrixflip(domc,'v')
    
    x0 = np.min(X);  y0 = np.min(Y);
    x1 = np.max(X);  y1 = np.max(Y);
    plt.imshow(domc, extent=[x0, x1, y0, y1])
    plt.show()
