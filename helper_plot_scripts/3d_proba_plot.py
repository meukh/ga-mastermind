from __future__ import division
import numpy as np
from deap import base, tools, creator
from functools import partial
import pickle
from os import path, sys
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
#from IPython import embed


## Starting the Game
## Drawing a random code
# variables to be set accoring to pickle file name
NB_COLORS=30
eaLambda= 2
eaMu= 1
NB_SLOTS=30

## Evaluating guesses against the master code
def evalGuess(guess, code):
    if len(guess) != NB_SLOTS:
        raise Exception("evalGuess expects guess of length %d."%NB_SLOTS)
    for color in guess:
        if color not in range(NB_COLORS):
            raise Exception("Unknown color %s."%color)

    diff = code - np.array(guess)
    return len(diff[diff!=0])

# Container of individual guesses
creator.create('distanceMin', base.Fitness, weights=(-1.0,))
creator.create('Guess', np.ndarray, fitness=creator.distanceMin)
    
## EA toolbox
eaBox = base.Toolbox()
eaBox.register('guess', tools.initIterate, creator.Guess, partial(np.random.choice, NB_COLORS, NB_SLOTS ))
eaBox.register('population', tools.initRepeat, list, eaBox.guess)

# Description: Reads list of saved log files from arguments, computes probability P(Quality,Time) from extracted data,
# then plots a 3d-surface where X=Time/Generations, Y=Fitness-values, Z=P(X, Y).

## PLOT
## Comparison PLOT (nubdha)
### Extracting records
logs = []
for pickle_file in sys.argv[1:]:
    pickled_log = open(pickle_file, 'rb')
    logs.append(pickle.load(pickled_log))
    pickled_log.close()

# Getting the longest running time among records
maxRT = 0
red_logs = {}
for i,log in enumerate(logs):
    red_logs["run_%d"%i] = {'best_fit':np.array(log.select('min')), 'gen':np.array(log.select('generation'))}
    maxRT = red_logs['run_%d'%i]['gen'][-1] if red_logs['run_%d'%i]['gen'][-1] > maxRT else maxRT


## Def probability(Quality, RunTime) 
def proba(q, t, logs):
    pb = 0
    for _, run_stat in logs.iteritems():
        idxs = np.where((run_stat['best_fit']!=None)&(run_stat['best_fit'] <= q))[0]
        if idxs.size>0 and idxs[0] <= t:
            pb += 1
    return pb/len(red_logs)


X, Y = np.arange(NB_SLOTS), np.arange(100, maxRT, 100)
Z = np.array([[proba(q,t, red_logs) for q in X] for t in Y])
X, Y = np.meshgrid(X, Y)

#embed()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.winter, label='GA on k,n Mastercode')
ax.set_xlabel('Fitness (Best Fitness is 0)')
ax.set_ylabel('Runtime (number of genrations)')
ax.set_zlabel('Probability')
plt.xlim(NB_SLOTS, 0)
plt.show()
