import pandas
import numpy as np
import scipy.sparse.linalg as linalg
import time

max_iter = 100
trials = 1000
convergence_tol = 10**-4

lam = 1.0/100.0

# Define all the optimization methods needed for project

def StochNewton(fun, grad, hess, dat, x0, bsz ):
    # fun, grad  and hess must each accept a vector x and data as input. Fun returns a real number, grad returns a vector, hess returns a matrix
    t0 = time.time()
    gam = 0.9
    jmax = int(np.ceil(np.log(10**-14)/np.log(gam)))
    eta = 0.5
    CGimax = 20 
    n = len(dat[:,0])
    bsz = min(n,bsz)
    
    
    f_rec = np.zeros(max_iter+1)
    f_rec[0]=fun(x0, dat)
    
    norm_grad_rec = np.zeros(max_iter)
    nfail = 0
    nfail_max = 5*np.ceil(n/bsz)
    ts = np.zeros(max_iter)

    x = x0
    for k in range(max_iter):
        ig = np.random.permutation(n)[0:bsz]
        ih = np.random.permutation(n)[0:bsz]
        datg = dat[ig,:]
        dath = dat[ih,:]
        
        hv = lambda v: hessv(v,x,dath)        
        app_hess = linalg.LinearOperator(shape=(len(x0),len(x0)), matvec = hv)
        app_grad = grad(x, datg)
        norm_grad_rec[k]=np.linalg.norm(app_grad)
        step = linalg.cg(app_hess, -1*app_grad, maxiter = CGimax)[0]
        a=1
        f0 = fun(x, datg)
        junk = eta*np.dot(app_grad,step)
        
        attempt = 0
        for j in range(jmax):
            attempt = j+1
            xtry = x+a*step
            f1 = fun(xtry, datg)
            if f1<f0+a*junk:
                break
            else:
                a=a*gam
        if attempt<jmax:
            x = xtry
        else:
            nfail=nfail+1
        
        f_rec[k+1]=fun(x,dat)
        
        ts[k] = time.time()-t0
        
                
        if nfail==nfail_max:            
            f_rec[k+2:]=f_rec[k+1]
            norm_grad_rec[k+1:]=norm_grad_rec[k]
            break
    
    return (x, f_rec, norm_grad_rec, k, ts)

def StochGradDescent(fun, grad, dat, x0, stepMethod, bsz):
    t0 = time.time()
    n = len(dat[:,0])
    bsz = min(n, bsz)
    
    # print(x0)
    x = x0
    
    f_rec = np.zeros(max_iter+1)
    f_rec[0]=fun(x0, dat)
    
    norm_grad_rec = np.zeros(max_iter)
    ts = np.zeros(max_iter)
        
    for k in range(max_iter):
        inds = np.random.permutation(np.arange(n))[:bsz]
        subdat=dat[inds,:]
        g = grad(x, subdat)
        a = stepsize(x, -g, g, fun, k, stepMethod, subdat)        
        x = x-a*g  
        # print('x '+str(x))
        
        f_rec[k+1] = fun(x,dat)
        # print('f '+str(f_rec[k+1])+'\n')
        norm_grad_rec[k] = np.linalg.norm(g)
        
        if k>=4:
            m = np.max(norm_grad_rec[k-4:k])
            if m<convergence_tol:
                break
        
        ts[k] = time.time()-t0
    
    f_rec[k+2:]=f_rec[k+1]
    norm_grad_rec[k+1:]=norm_grad_rec[k]
    
    return (x, f_rec, norm_grad_rec, k, ts)
    

def SLBFGS(fun, grad, dat, x0, stepMethod, bsz_g, bsz_h, M):
    t0 = time.time()
    n = len(dat[:,0])
    m = 5 # limited memory constant
    dim = len(x0)
    s = np.zeros((m,dim))
    y= np.zeros((m,dim))
   
    bsz_g = min(n, bsz_g)
    bsz_h = min(n, bsz_h)
    
    
    x = x0
    # print('x '+str(x))
    
    ig = np.random.permutation(n)[0:bsz_g]
    datg = dat[ig,:]  
    g = grad(x0,datg)
        
    f_rec = np.zeros(max_iter+1)
    f_rec[0]=fun(x0, dat)    
    norm_grad_rec = np.zeros(max_iter)
    ts = np.zeros(max_iter)
    
    # take first step
    x = x-g
    # print('x '+str(x)+'\n')
    
    s[0] = -g
    
    ig = np.random.permutation(n)[0:bsz_g]
    datg = dat[ig,:]  
    gnew = grad(x, datg)
    
    y[0] = gnew - g
    g= gnew   
    norm_grad_rec[0] = np.linalg.norm(g)
    f_rec[1] = fun(x, dat)
    
    for k in range(1,max_iter):
        ig = np.random.permutation(n)[0:bsz_g]
        datg = dat[ig,:]          
        
        if k<m: p = SLBFGS_dir(g, s[:k], y[:k])
        else: p = SLBFGS_dir(g, s, y)
        a = stepsize(x, p, g, fun, k, stepMethod, datg)
                              
        xnew = x+a*p                      

        if (k%M==0 or k<m):
            ih = np.random.permutation(n)[0:bsz_g]
            dath = dat[ih,:]
            s = np.roll(s, 1, axis=0)
            y = np.roll(y, 1, axis=0)
            s[0] = a*p
            y[0] = grad(xnew, dath) - grad(x, dath)
             
        x = xnew
        g = grad(xnew, datg) 
        
        f_rec[k+1] = fun(x,dat)
        # print("x "+str(x))
        # print('f '+str(f_rec[k+1]))
        # print(' ')
        norm_grad_rec[k] = np.linalg.norm(g)
        ts[k] = time.time()-t0
        
        if k>=4:
            mgrad = np.max(norm_grad_rec[k-4:k])
            if mgrad<convergence_tol:
                break 
        
    f_rec[k+2:]=f_rec[k+1]
    norm_grad_rec[k+1:]=norm_grad_rec[k]    
    
    return (x, f_rec, norm_grad_rec, k, ts)
        
        
def SLBFGS_dir(g, s, y):
    m = len(s[:,0])    
    rho = [ 1/np.dot(s[i],y[i]) for i in range(m) ]
    a = np.zeros(m)
    
    for i in range(m):
        a[i] = rho[i] * np.dot(s[i], g)
        g = g-a[i]*y[i]
        
    gam = np.dot(s[0],y[0])/np.dot(y[0],y[0])
    g = g*gam
    
    for i in range(m):
        j = m-i-1
        aux = rho[j] * np.dot(y[j],g)
        g = g + (a[j]-aux)*s[j]
    
    return -g        

    
    
def stepsize(x, d, g, fun, k, stepMethod, dat ):
    if stepMethod =='backsearch':
        a = 1
        r = 0.9
        e = np.dot(d,g)
        f0 = fun(x,dat)
        for j in range(300):
            xtry = x+a*d
            f1 = fun(xtry, dat)
            if f1<f0+0.5*a*e:
                break
            else:
                a=a*r                
        return a
    elif stepMethod =='harmonic':
        return 1/(k+1)
    elif stepMethod =='binary':
        p = int(np.ceil(np.log2(k+1)))
        if p==0: return 1
        else: return 2**(-p+1)
    else: return 1        
            


# set up objective function and related functions   
def fun(x, dat):
    d0 = len(dat[:,0])
    aux = -1*np.dot(dat,x)
    return np.sum(np.log(1+np.exp(aux)))/d0 + lam*np.dot(x,x)/2

def grad(x, dat):
    d0 = len(dat[:,0])
    d1 = len(dat[0,:])
    aux = np.exp(-1*np.dot(dat,x))
    mat = np.outer(aux/(1+aux),np.ones(d1))
    return np.sum(-1*dat*mat,0)/d0 + lam*x

def hessv(v, x, dat): 
    d1 = len(dat[0,:])
    aux = np.exp(-1*np.dot(dat,x))
    vec = (aux * np.dot(dat,v))/np.square(1+aux)
    mat = dat * np.outer(vec,np.ones(d1))
    return np.sum(mat,0) + lam*v


# set up data
def get_data(dset, year, rescale):
    # Read in the data for the correct year
    if year==16:
        raw = pandas.read_csv('/home/max/Desktop/AMSC/Project/A2016.csv').values
    else:
        raw = pandas.read_csv('/home/max/Desktop/AMSC/Project/A2012.csv').values
        
    # blabel is True if dem victory, False if repub victory
    # nlabel is 1 or -1
    blabels = np.greater(raw[:,3],raw[:,4])
    nlabels = 2*blabels.astype(int)-1 
    
    # pick the indices for the desired dataset
    if dset=='cal':
        indices = np.where( raw[:,2] ==' CA')[0]
    elif dset=='cal+':
        indices_ca = np.where( raw[:,2] ==' CA')[0]
        indices_or = np.where( raw[:,2] ==' OR')[0]
        indices_wa = np.where( raw[:,2] ==' WA')[0]
        indices = np.append(indices_ca,np.append(indices_or,indices_wa))
    else:
        dem_indices = np.where(blabels)[0]
        l = len(dem_indices)
        rep_indices = np.where(np.logical_not(blabels))[0]
        np.random.seed(seed=10)
        rep_indices = np.random.permutation(rep_indices)[0:l]
        indices = np.append(dem_indices,rep_indices)
        np.random.seed(time.time_ns() % (2**32 - 1))
        
    
    # take only the desired data from the raw
    raw = raw[indices]
    n = len(indices)
    
    blabels = blabels[indices]
    nlabels = nlabels[indices]
       

    votes = raw[:,3]+raw[:,4]
    log_votes = np.log( votes.astype(float) )
    vote_scale = np.max( log_votes )
    if rescale:
        log_votes = log_votes/vote_scale
    
    income = raw[:,5].astype(float)
    income_scale = np.max(income)
    if rescale:
        income = income/income_scale
    
    education = raw[:,9].astype(float)
    education_scale = np.max(education)
    if rescale:
        education = education/education_scale
    
    migra = raw[:,6].astype(float)
    migra_scale = np.max(np.abs(migra))
    if rescale:
        migra = migra/migra_scale
        

    data = np.transpose([ income, education, migra, np.ones(n) ])    
    for i in range(n):
        data[i] = data[i]*nlabels[i]   
        
    scales = {'Migration' : migra_scale, 'Income' : income_scale, 'Education' : education_scale}
    
    return (data,scales)


    