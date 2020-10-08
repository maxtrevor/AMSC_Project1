import numpy as np
import project_methods as pmeths
import pickle

def prob2():
    # get the objective function and related function from project methods
    fun = pmeths.fun
    grad = pmeths.grad
       
    max_i = pmeths.max_iter
    trials = pmeths.trials
    
    (data, scales) = pmeths.get_data('full', 12, True)
    
    stepMethods = ['harmonic', 'binary', 'backsearch']
    batchSizes = [64, 128, 256]
    combos = [[a,b] for a in range(3) for b in range(3)]
    
    xs = np.zeros((3, 3, trials, 4))
    f_recs = np.zeros((3, 3, trials, max_i+1))
    norm_recs = np.zeros((3,3, trials, max_i))
    ks = np.zeros((3,3, trials))
    ts = np.zeros((3,3, trials, max_i))
    
    x0 = np.array([0,0,0,0])
    
    for c in combos:
            print(str(c)+' prob 2')
            for i in range(trials):
                (x, f_rec, norm_grad_rec,k,t_rec) = pmeths.StochGradDescent(fun, grad, data, x0, stepMethods[c[0]], batchSizes[c[1]])
                
                xs[c[0],c[1],i]=x
                f_recs[c[0],c[1],i] = f_rec
                norm_recs[c[0],c[1],i] = norm_grad_rec
                ks[c[0],c[1],i] = k
                ts[c[0],c[1],i] = t_rec
                
    xs_out = np.average(xs, axis=2)
    f_recs_out = np.average(f_recs, axis=2)
    norm_recs_out = np.average(norm_recs, axis=2)
    ks_out = np.average(ks, axis=2)
    ts_out = np.average(ts, axis=2)
    
    results = {'x':xs_out, 'f':f_recs_out, 'n':norm_recs_out, 'k': ks_out, 't':ts_out}
    pickle.dump( results, open( "res2.p", "wb" ) )