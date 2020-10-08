import numpy as np
import project_methods as pmeths
import matplotlib.pyplot as plt
import pickle
import time
import project_prob1 as p1
import project_prob2 as p2
import project_prob3 as p3
import project_prob4 as p4

t_start = time.time()

# controls: set something to 1 to have it happen, 0 to skip
doProb1 = 0
doProb2 = 0
doProb3 = 0
doProb4 = 0

plot2 = 0
plot3 = 0
plot4 = 1

if doProb1: p1.prob1()

print('starting prob 2')
if doProb2: p2.prob2()
# plot results of problem 2

if plot2:
    res2 = pickle.load( open( "res2.p", "rb" ) )
    stepMethods = ['harmonic', 'binary', 'backsearch']
    batchSizes = [64, 128, 256]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    combos = [[a,b] for a in range(3) for b in range(3)]
    for c in combos:
        k = res2['k'][c[0],c[1]]
        lab =  stepMethods[c[0]]+' '+str(batchSizes[c[1]])
        ax.plot(np.arange(k), res2['f'][c[0], c[1], 0:int(k)], label=lab)
    
    ax.legend() 
    plt.show()    

print('starting prob 3')
if doProb3: p3.prob3()

#plot results of problem 3
if plot3:
    res3 = pickle.load( open( "res3.p", "rb" ) )
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for (b, bsz) in enumerate(batchSizes):
        k = res3['k'][b]
        lab =  'bsz: '+str(bsz)
        ax.plot(np.arange(k), res3['f'][b, 0:int(k)], label=lab)
    
    ax.legend() 
    plt.show()    

print('starting prob 4')
if doProb4: p4.prob4()

# plot results of problem 4
if plot4:
    res4_1 = pickle.load( open( "res4_1.p", "rb" ) )
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for c in combos:
        k = res4_1['k'][c[0],c[1]]
        lab =  'bsz_g = '+str(batchSizes[c[0]])+' bsz_h = '+str(1+c[1])+'*bsz_g'
        ax.plot(np.arange(k), res4_1['f'][c[0], c[1], 0:int(k)], label=lab)
    
    ax.legend() 
    plt.show()  
    
    
    res4_2 = pickle.load( open( "res4_2.p", "rb" ) )
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for c in combos:
        k = res4_2['k'][c[0],c[1]]
        lab =  stepMethods[c[0]]+', bsz_g = '+str(batchSizes[c[1]])
        ax.plot(np.arange(k), res4_2['f'][c[0], c[1], 0:int(k)], label=lab)
    
    ax.legend() 
    plt.show()  
    
    res4_3 = pickle.load( open( "res4_3.p", "rb" ) )
    fig = plt.figure()
    ax = fig.add_subplot(111)
    M = [1, 5, 10, 20 ]
    
    for (i,M) in enumerate(M):
        k = res4_3['k'][i]
        lab =  'M = '+str(M)
        ax.plot(np.arange(k), res4_3['f'][i, 0:int(k)], label=lab)
    
    ax.legend() 
    plt.show()  
    
print(time.time()-t_start)