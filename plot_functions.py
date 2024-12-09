import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.animation import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import jv
from scipy.special import hankel1
from scipy.special import jn_zeros
from scipy.signal import argrelextrema
from scipy.sparse.linalg import eigsh
from numpy.linalg import norm
from numpy.random import *

from base_functions import *

###########################################################################
##    Plotting functions
###########################################################################

def plot_all_domains_from_file(filename, domainClass, param):
    file_ab = open(filename,'r')
    lines = file_ab.readlines()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    for i in range(len(lines)):
        fig.clear(True) 
        c = read_from_file(filename, i)
        domain = domainClass(c,param)
        domain.rescale_domain()               # rescale and redistribute collocation points
        domain.set_domain_variables()         # set collocation and source points
        
        Gamma_shifted_x = np.append(domain.Gamma_col[0,:], domain.Gamma_col[0,0])
        Gamma_shifted_y = np.append(domain.Gamma_col[1,:], domain.Gamma_col[1,0])
        #Gamma_shifted_x =  Gamma_shifted_x-domain.CoM[0]
        #Gamma_shifted_y =  Gamma_shifted_y-domain.CoM[1]
        
        plt.plot(Gamma_shifted_x,Gamma_shifted_y,'b')
        
        plt.axis('square')
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.grid(True)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(filename[-18:-7] + ", i=" + str(i))

        plt.pause(0.001)

    file_ab.close()
    plt.show()

def animate_domains_from_file(filename, domainClass, param):
    file_ab = open(filename,'r')
    lines = file_ab.readlines()

    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = ax.plot([], [], 'b')

    ax.set_aspect('equal')
    plt.title(filename)
    plt.axis('square')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")     

    def update(i):
        #fig.clear(True)
        c = read_from_file(filename, i)
        domain = domainClass(c,param)
        domain.rescale_domain()               # rescale and redistribute collocation points
        domain.set_domain_variables()         # set collocation and source points 
        Gamma_shifted_x = np.append(domain.Gamma_col[0,:], domain.Gamma_col[0,0])
        Gamma_shifted_y = np.append(domain.Gamma_col[1,:], domain.Gamma_col[1,0])
        Gamma_shifted_x =  Gamma_shifted_x-domain.CoM[0]
        Gamma_shifted_y =  Gamma_shifted_y-domain.CoM[1]   	
        #plt.plot(Gamma_shifted_x,Gamma_shifted_y,'b')
        ln.set_data(Gamma_shifted_x, Gamma_shifted_y)
        
        return ln

    ani = FuncAnimation(fig, update, frames=len(lines), repeat=True)#,  blit=True)#init_func=init,
    ani.save(filename[:-7] + "_anim.gif", fps=10)
    
    #plt.show()

    file_ab.close()
    
    
    
def plot_last_domain_from_file(filename, domainClass, param):
    file_ab = open(filename,'r')
    lines = file_ab.readlines()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    c = read_from_file(filename, -1)
    domain = domainClass(c,param)
    domain.rescale_domain()               # rescale and redistribute collocation points
    domain.set_domain_variables()         # set collocation and source points
    plt.plot(domain.Gamma_col[0,:],domain.Gamma_col[1,:],'b')
    
    plt.axis('square')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(filename)


    file_ab.close()
    plt.show()
    
    
    
def plot_all_domains_from_population_file(filename, domainClass, param):
    scal = [0.4758151, 0.8795453]
    file_ab = open(filename,'r')
    lines = file_ab.readlines()
    dom_per_row = np.floor(np.sqrt(len(lines)))
    fig, ax = plt.subplots()
    for i in range(len(lines)):
        c = read_from_file(filename, i)
        domain = domainClass(c,param)
        domain.rescale_domain()               # rescale and redistribute collocation points 
        domain.set_domain_variables()         # set collocation and source points
        plt.plot(domain.Gamma_col[0,:]+2*np.floor(i/dom_per_row),domain.Gamma_col[1,:] + 2*(i % dom_per_row))#,'b')
        #next(ax._get_lines.prop_cycler)
        #next(ax._get_lines.prop_cycler)
        #next(ax._get_lines.prop_cycler)
        #next(ax._get_lines.prop_cycler)
        #next(ax._get_lines.prop_cycler)
        #next(ax._get_lines.prop_cycler)
        #next(ax._get_lines.prop_cycler)
        #next(ax._get_lines.prop_cycler)
        #next(ax._get_lines.prop_cycler)
    plt.axis('square')
    #plt.grid(True)
    plt.xlim(-1,1) #plt.xlim(-0.5,1.5)
    plt.ylim(-1,1)
    plt.axis("off")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
    
    
def plot_all_domains_from_population(A_pop, B_pop, param):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    size_pop = np.shape(A_pop)

    dom_per_row = np.floor(np.sqrt(size_pop[0]))

    for i in range(size_pop[0]):
        a = A_pop[i,:][np.newaxis]
        b = B_pop[i,:][np.newaxis]
        domain = Domain(a,b,param)
        domain.rescale_domain()               # rescale and redistribute collocation points 
        domain.set_domain_variables()         # set collocation and source points
        plt.plot(domain.Gamma_col[0,:]+2*(i % dom_per_row),domain.Gamma_col[1,:] + 2*np.floor(i/dom_per_row),'b')
        
    plt.axis('square')
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
    
# plot single domain, with rotation/centering
def plot_domain(domain, param, rot_angles, filename):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    Gamma_shifted_x = np.append(domain.Gamma_col[0,:], domain.Gamma_col[0,0])
    Gamma_shifted_y = np.append(domain.Gamma_col[1,:], domain.Gamma_col[1,0])
    Gamma_shifted_x =  Gamma_shifted_x-domain.CoM[0]
    Gamma_shifted_y =  Gamma_shifted_y-domain.CoM[1]
 
    for phi in rot_angles:
        Gamma_rot_x = np.cos(phi)*Gamma_shifted_x-np.sin(phi)*Gamma_shifted_y
        Gamma_rot_y = np.sin(phi)*Gamma_shifted_x+np.cos(phi)*Gamma_shifted_y
        plt.plot(Gamma_rot_x,Gamma_rot_y)
        #plt.plot(domain.p_source[0,:],domain.p_source[1,:],'r.')
        #plt.plot(domain.Gamma_interior[0,:],domain.Gamma_interior[1,:],'k.')
        
    plt.axis('square')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    
    plt.axis("off")
    #plt.grid(True)
    
    plt.xlabel("x")
    plt.ylabel("y")
    #plt.title(filename)
    
    if filename != None:
        plt.savefig(filename)
    plt.show()


def plot_final_domains_from_files(param, file_list, domainClass, angles, writeToFileBool):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    for k in range(len(file_list)):
        file_name = file_list[k]
        phi = angles[k]

        c = read_from_file(file_name, -1)
        domain = domainClass(a,b,param)
        domain.rescale_domain()               # rescale and redistribute collocation points
        domain.set_domain_variables()         # set collocation and source points

        Gamma_shifted_x = np.append(domain.Gamma_col[0,:], domain.Gamma_col[0,0])
        Gamma_shifted_y = np.append(domain.Gamma_col[1,:], domain.Gamma_col[1,0])
        Gamma_shifted_x =  Gamma_shifted_x-domain.CoM[0]
        Gamma_shifted_y =  Gamma_shifted_y-domain.CoM[1]
        Gamma_rot_x = np.cos(phi)*Gamma_shifted_x-np.sin(phi)*Gamma_shifted_y
        Gamma_rot_y = np.sin(phi)*Gamma_shifted_x+np.cos(phi)*Gamma_shifted_y

        if writeToFileBool == 1:
            writeXYToTxt(file_name[:-7],Gamma_rot_x,Gamma_rot_y)

        Gamma_rot_x = Gamma_rot_x +2*(k % 5)
        Gamma_rot_y = Gamma_rot_y -2*np.floor(k/5)
        plt.plot(Gamma_rot_x,Gamma_rot_y,'b')
    
 
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def get_final_domains_curve_from_files(Bs, file_list, file_out):

    #fig = plt.figure()
    J_curve = np.zeros((len(Bs),))

    print("loading files:")
    for k in range(len(Bs)):
        file_name = file_list[k]
        print("i = " + str(k) + ": " + file_name)

        a, _ = read_from_file_val(file_name, -1) 
        J_curve[k] = a

    if file_out != None :
        writeXYToTxt(file_out,Bs,J_curve)

    return J_curve


def get_diamagnetism_curve_from_files(n, objType, filename):
    
    file_ab = open(filename,'r')
    lines = file_ab.readlines()
    Nb = len(lines)
    
    bs_disc = np.zeros((Nb,))
    J_disc_curve = np.zeros((Nb,))
    for k in range(Nb):
        B, a = read_from_file_val(filename, k) 
        bs_disc[k] = B
        
        if objType == 'sum':
            J_disc_curve[k] = np.sum(a[:n])   
        elif objType == 'lambda':
            J_disc_curve[k] = np.sum(a[n-1])
        else:
            print("Error: unknown objective type!")
            
    print(J_disc_curve)
    
    return bs_disc, J_disc_curve






























#################################################################
########### UTILITY FUNCTIONS ###################################
#################################################################

def plot_disc_eigenfunctions(param):
    a = np.zeros((1,param.len_ab))
    b = np.zeros((1,param.len_ab))
    a[0,0:2] = np.array([1.0, 0.0])

    sum_n, k_crit, mult_n, V = direct_problem(a,b,param,1)
    x=np.arange(-1,1,dh)
    y=np.arange(-1,1,dh)
    X,Y = np.meshgrid(x,y)
    #Z = eigf(X,Y,kappa,alpha,p_source)

    #for m in range(1):

    fig = plt.figure()
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot_surface(X, Y, np.abs(Z**2), cmap='plasma', edgecolor='none')
    #ax.set_title('Eigenfunction')
    #plt.show()



## compute first n eigenvalues and sum(n-sum_length+1:n) of disc 
#
def disc_eigenvalues(n):
    eigenvalues = jn_zeros(0,n)
    for m in range(1,n):
        eigenvalues=np.append(eigenvalues,jn_zeros(m,n))
        eigenvalues=np.append(eigenvalues,jn_zeros(m,n))    # double eigenvalues!
    eigenvalues = np.sort(eigenvalues)
    eigenvalues = np.pi*eigenvalues**2
    

    sums = np.zeros(n)
    for m in range(n):
        sums[m] = np.sum(eigenvalues[:(m+1)])

    return eigenvalues, sums

def print_disc_eigenvalues(n):

    eigenvalues, sums = disc_eigenvalues(n)

    print("Eigenvalues of disc:")
    print(eigenvalues[:n])
    print("Eigenvalue sums:")
    print(sums[:n])



## plot eigenfunctions of disc ( abs^2 )
#
def plot_disc_eigenfunction(n, sort_end):
    eigenvalues = np.zeros((n,n))
    eigenvalues[0,:] = jn_zeros(0,n)
    for m in range(1,n):
        eigenvalues[m,:]=jn_zeros(m,n)          # double eigenvalues!
    
    dh = 0.01
    x=np.arange(-1,1,dh)
    y=np.arange(-1,1,dh)
    t=np.arange(0,2*np.pi,dh)
    X,Y = np.meshgrid(x,y)
    R = np.sqrt(X**2+Y**2)
    T = np.arccos(X/R)
    T[np.less(Y,0)] = 2*np.pi - T[np.less(Y,0)]
    circ_x = np.cos(t)
    circ_y = np.sin(t)

    fig = plt.figure()

    #for k in range(n):
    #    Z = jv(0,eigenvalues[0,k]*R)
    #    plt.pcolormesh(X, Y+2*k, np.abs(Z**2), cmap='plasma', edgecolor='none')
    #    plt.plot(circ_x,circ_y+2*k,'r')
    #for m in range(1,n):
    #    for k in range(n):
    #        Z1 = jv(m,eigenvalues[m,k]*R)*np.sin(m*T)
    #        Z2 = jv(m,eigenvalues[m,k]*R)*np.cos(m*T)
    #        plt.pcolormesh(X+2*m, Y+2*k, np.abs(Z1**2), cmap='plasma', edgecolor='none')
    #        plt.plot(circ_x+2*m,circ_y+2*k,'r')
    
    eigenvalues_=eigenvalues.reshape((n**2, 1))
    sort_ind = np.argsort(eigenvalues_, axis=0)
    
    sort_ind_m = sort_ind[:,0] //n
    sort_ind_k = sort_ind[:,0] % n

    for j in range(sort_end):
        Z2 = jv(sort_ind_m[j],eigenvalues[sort_ind_m[j],sort_ind_k[j]]*R)*np.cos(sort_ind_m[j]*T)
        plt.pcolormesh(X+2*(j), Y, np.abs(Z2**2), cmap='plasma', edgecolor='none')
        plt.plot(circ_x+2*(j),circ_y,'r')
        if sort_ind_m[j] > 0:
            Z1 = jv(sort_ind_m[j],eigenvalues[sort_ind_m[j],sort_ind_k[j]]*R)*np.sin(sort_ind_m[j]*T)
            plt.pcolormesh(X+2*(j), Y - 2, np.abs(Z1**2), cmap='plasma', edgecolor='none')
            plt.plot(circ_x+2*(j),circ_y - 2,'r')
        
    plt.show()

## write x,y column pair to txt file for Latex plots
#
def writeXYToTxt(filename,x,y):
    file_out_xy = filename + "_xy.txt"

    shape_x = np.shape(x)

    # write x,y columns to file
    f_xy = open(file_out_xy,'a')
    for i in range(shape_x[0]):                    # x,y pairs
        s = format(x[i], '.3f')+", "+format(y[i], '.3f')
        f_xy.write(s+'\n')
    f_xy.close()
