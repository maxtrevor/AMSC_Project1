import numpy as np
import project_methods as pmeths
import pickle

def prob4():
        # get the objective function and related function from project methods
    fun = pmeths.fun
    grad = pmeths.grad
    hessv = pmeths.hessv
       
    max_i = pmeths.max_iter
    trials = pmeths.trials
    
    (data, scales) = pmeths.get_data('full', 12, True)
    
    batchSizes = [64, 128, 256]
    stepMethods = ['harmonic', 'binary', 'backsearch']
    freqs = [1, 5, 10, 20 ]
    
    combos = [[a,b] for a in range(3) for b in range(3)]
    
    xs_test1 = np.zeros((3, 3, trials, 4))
    f_recs_test1 = np.zeros((3, 3, trials, max_i+1))
    norm_recs_test1 = np.zeros((3,3, trials, max_i))
    ks_test1 = np.zeros((3,3, trials))
    ts_test1 = np.zeros((3,3, trials, max_i))
    
    x0 = np.array([0,0,0,0])
    x_opt = x0
    f_opt = 100
    
    print('starting prob 4 1')
    for c in combos:
        print(str(c)+' prob 4_1')
        for i in range(trials):
            bszg = batchSizes[c[0]]
            (x, f_rec, norm_grad_rec,k,t_rec) = pmeths.SLBFGS(fun, grad, data, x0, 'backsearch', bszg, (c[1]+1)*bszg, 5 )
            
            
            if f_rec[-1] < f_opt:
                f_opt = f_rec[-1]
                x_opt = x
            
            xs_test1[c[0],c[1],i]=x
            f_recs_test1[c[0],c[1],i] = f_rec
            norm_recs_test1[c[0],c[1],i] = norm_grad_rec
            ks_test1[c[0],c[1],i] = k
            ts_test1[c[0],c[1],i] = t_rec
    
    xs_out1 = np.average(xs_test1, axis=2)
    f_recs_out1 = np.average(f_recs_test1, axis=2)
    norm_recs_out1 = np.average(norm_recs_test1, axis=2)
    ks_out1 = np.average(ks_test1, axis=2)
    ts_out1 = np.average(ts_test1, axis=2)
    
    
    
    results1 = {'x':xs_out1, 'f':f_recs_out1, 'n':norm_recs_out1, 'k': ks_out1, 't':ts_out1}
    pickle.dump( results1, open( "res4_1.p", "wb" ) )
            
    xs_test2 = np.zeros((3, 3, trials, 4))
    f_recs_test2 = np.zeros((3, 3, trials, max_i+1))
    norm_recs_test2 = np.zeros((3,3, trials, max_i))
    ks_test2 = np.zeros((3,3, trials))
    ts_test2 = np.zeros((3,3, trials, max_i))
    
    print('starting prob 4 2')
    for c in combos:
        print(str(c)+' prob 4_1')
        for i in range(trials):
            bszg = batchSizes[c[1]]
            (x, f_rec, norm_grad_rec,k,ts) = pmeths.SLBFGS(fun, grad, data, x0, stepMethods[c[0]], bszg, 2*bszg, 5 )
                        
            if f_rec[-1] < f_opt:
                f_opt = f_rec[-1]
                x_opt = x
            
            xs_test2[c[0],c[1],i]=x
            f_recs_test2[c[0],c[1],i] = f_rec
            norm_recs_test2[c[0],c[1],i] = norm_grad_rec
            ks_test2[c[0],c[1],i] = k
            ts_test2[c[0],c[1],i] = ts
            
        
    xs_out2 = np.average(xs_test2, axis=2)
    f_recs_out2 = np.average(f_recs_test2, axis=2)
    norm_recs_out2 = np.average(norm_recs_test2, axis=2)
    ks_out2 = np.average(ks_test2, axis=2)
    ts_out2 = np.average(ts_test2, axis=2)
    
    results2 = {'x':xs_out2, 'f':f_recs_out2, 'n':norm_recs_out2, 'k': ks_out2, 't':ts_out2}
    pickle.dump( results2, open( "res4_2.p", "wb" ) )
    
    xs_test3 = np.zeros((4, trials, 4))
    f_recs_test3 = np.zeros((4, trials, max_i+1))
    norm_recs_test3 = np.zeros((4, trials, max_i))
    ks_test3 = np.zeros((4, trials))
    ts_test3 = np.zeros((4, trials, max_i))
    
    print('starting prob 4 3')
    for a in range(4):
        print(str(a)+' prob 4_1')
        for i in range(trials):
            (x, f_rec, norm_grad_rec, k, ts) = pmeths.SLBFGS(fun, grad, data, x0, 'backsearch', 256, 512, freqs[a] )
                        
            if f_rec[-1] < f_opt:
                f_opt = f_rec[-1]
                x_opt = x
            
            xs_test3[a, i]=x
            f_recs_test3[a, i] = f_rec
            norm_recs_test3[a, i] = norm_grad_rec
            ks_test3[a, i] = k
            ts_test3[a, i] = ts
            
    xs_out3 = np.average(xs_test3, axis=1)
    f_recs_out3 = np.average(f_recs_test3, axis=1)
    norm_recs_out3 = np.average(norm_recs_test3, axis=1)
    ks_out3 = np.average(ks_test3, axis=1)
    ts_out3 = np.average(ts_test3, axis=1)
    
    results3 = {'x':xs_out3, 'f':f_recs_out3, 'n':norm_recs_out3, 'k': ks_out3, 't':ts_out3}
    pickle.dump( results3, open( "res4_3.p", "wb" ) )
    
    print('f opt: '+str(f_opt))
    print('x_opt: '+str(x_opt))