import numpy as np
import project_methods as pmeths
import pickle

def prob3():
    # get the objective function and related function from project methods
    fun = pmeths.fun
    grad = pmeths.grad
    hessv = pmeths.hessv
       
    max_i = pmeths.max_iter
    trials = pmeths.trials
    
    (data, scales) = pmeths.get_data('full', 12, True)
    
    batchSizes = [64, 128, 256]
    
    xs = np.zeros((3, trials, 4))
    f_recs = np.zeros((3, trials, max_i+1))
    norm_recs = np.zeros((3, trials, max_i))
    ks = np.zeros((3, trials))
    ts = np.zeros((3, trials, max_i ))
    
    x0 = np.array([0,0,0,0])
    
    for (b, bsz) in enumerate(batchSizes):
        print(str(b)+' prob 3')
        for i in range(trials):            
            (x, f_rec, norm_grad_rec,k,t_rec) = pmeths.StochNewton(fun, grad, hessv, data, x0, bsz)            
            xs[b, i]=x
            f_recs[b, i] = f_rec
            norm_recs[b, i] = norm_grad_rec
            ks[b, i] = k
            ts[b, i] = t_rec
            
    xs_out = np.average(xs, axis=1)
    f_recs_out = np.average(f_recs, axis=1)
    norm_recs_out = np.average(norm_recs, axis=1)
    ks_out = np.average(ks, axis=1)
    ts_out = np.average(ts, axis=1)
    
    results = {'x':xs_out, 'f':f_recs_out, 'n':norm_recs_out, 'k': ks_out, 't':ts_out}
    pickle.dump( results, open( "res3.p", "wb" ) )