
import numpy as np
import project_methods as pmeths
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds
from mpl_toolkits.mplot3d import Axes3D


    
def prob1():
    # get the objective function and related function from project methods
    fun = pmeths.fun
    grad = pmeths.grad
    hessv = pmeths.hessv
    
    # use stochastic inexact newton's for initial guess
    (cal_dat, scales) = pmeths.get_data('cal',12, True)
    n = len(cal_dat[:,0])
    d = 3
    labels = cal_dat[:,3]
    
    x0 = np.random.random(4)*2-1
    (x0, f_rec, norm_grad_rec, kmax, t0)= pmeths.StochNewton(fun,grad,hessv,cal_dat,x0, 64)
    print('Stoch Newton Output: '+str(x0))
    
    # get expanded data set for pacific northwest
    (pnw_dat, junk) = pmeths.get_data('cal+', 12, False)
    n2 = len(pnw_dat[:,0])
    pnw_labels = pnw_dat[:,3]
    
    # set up soft margin variables for consrained optimization
    xi = np.array([max(0,1-np.dot(cal_dat[i,:],x0)) for i in range(n)])+10**-15
    x1 = np.append(x0,xi)
    
    #perform constrained minimization
    hess = np.block([ [np.eye(d), np.zeros((d,n+1))] , [np.zeros((n+1,d)),np.zeros((n+1,n+1))]])
    offset = np.block([np.zeros(d+1), np.ones(n)])
    
    def fun(v):
        return np.dot(v,np.dot(hess,v))/2+1000*np.dot(offset,v)
    
    def grad(v):
        return np.dot(hess,v)+1000*offset
    
    mat = np.block([[cal_dat, np.eye(n)],[np.zeros((n,d+1)),np.eye(n)]])
    lcon = np.append(np.ones(n),np.zeros(n))
    ucon = np.inf*np.ones(2*n)
    cons = LinearConstraint(mat, lcon, ucon, keep_feasible=True)
    lb = np.append(-np.inf*np.ones(d+1),np.zeros(n))
    ub = np.inf*np.ones(n+d+1)
    bs = Bounds(lb,ub, keep_feasible=True)
    out = minimize(fun, x1, method='SLSQP', jac=grad, constraints=cons, tol = 10**-16, bounds=bs)
    x = out['x']

    print('Constrained min '+str(x))

    # test performance of the two planes
    cal_test0 = np.greater(np.dot(cal_dat,x0), np.zeros(n) ).astype(int)
    cal_test0_result = np.sum(cal_test0)/n
    
    cal_test1 = np.greater(np.dot(cal_dat,x[:4]), np.zeros(n) ).astype(int)
    cal_test1_result = np.sum(cal_test1)/n
    
    pnw_test0 = np.greater(np.dot(pnw_dat,x0), np.zeros(n2) ).astype(int)
    pnw_test0_result = np.sum(pnw_test0)/n2
    
    pnw_test1 = np.greater(np.dot(pnw_dat,x[:4]), np.zeros(n2) ).astype(int)
    pnw_test1_result = np.sum(pnw_test1)/n2
    
    results = np.array([cal_test0_result, cal_test1_result, pnw_test0_result, pnw_test1_result])
    print(results)
    
    
    
    # Code to plot California data    
    dem_indices = np.where(labels==1)[0]
    rep_indices = np.where(labels==-1)[0]
    
    dem_dat = cal_dat[dem_indices]
    rep_dat = -1*cal_dat[rep_indices]
    
    
    xs = np.linspace(0,1,10)
    ys = np.linspace(0,1,10)
    X,Y = np.meshgrid(xs,ys)
    Z=(-x[3]-x[0]*X-x[1]*Y)/x[2]
    Z1 = (-x1[3]-x1[0]*X-x1[1]*Y)/x1[2]
    
    # plot cal_dat without plane
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    
    ax.scatter( dem_dat[:,0], dem_dat[:,1], dem_dat[:,2] )
    ax.scatter( rep_dat[:,0], rep_dat[:,1], rep_dat[:,2] )
    ax.set_xlabel('Income')
    ax.set_ylabel('Education')
    ax.set_zlabel('Migration')
    
    # plot cal_dat with initial plane
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111,projection='3d')
    ax2.plot_surface(X,Y,Z1)
    
    ax2.scatter( dem_dat[:,0], dem_dat[:,1], dem_dat[:,2] )
    ax2.scatter( rep_dat[:,0], rep_dat[:,1], rep_dat[:,2] )
    ax2.set_xlabel('Income')
    ax2.set_ylabel('Education')
    ax2.set_zlabel('Migration')
    
    # plot cal_dat with optimized plane
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111,projection='3d')
    ax3.plot_surface(X,Y,Z)
    
    ax3.scatter( dem_dat[:,0], dem_dat[:,1], dem_dat[:,2] )
    ax3.scatter( rep_dat[:,0], rep_dat[:,1], rep_dat[:,2] )
    ax3.set_xlabel('Income')
    ax3.set_ylabel('Education')
    ax3.set_zlabel('Migration')
    
    
    
    
    
    
    # plot pnw dat
    pnw_dem_indices = np.where(pnw_labels==1)[0]
    pnw_rep_indices = np.where(pnw_labels==-1)[0]
    pnw_dem_dat = pnw_dat[pnw_dem_indices]
    pnw_rep_dat = -1*pnw_dat[pnw_rep_indices]
    
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111,projection='3d')
    ax4.plot_surface(X,Y,Z)
    
    # rescale pnw data to same scale as california
    ax4.scatter( pnw_dem_dat[:,0]/scales['Income'], pnw_dem_dat[:,1]/scales['Education'], pnw_dem_dat[:,2]/scales['Migration'] )
    ax4.scatter( pnw_rep_dat[:,0]/scales['Income'], pnw_rep_dat[:,1]/scales['Education'], pnw_rep_dat[:,2]/scales['Migration'] )
    ax4.set_xlabel('Income')
    ax4.set_ylabel('Education')
    ax4.set_zlabel('Migration')
    
    plt.show()