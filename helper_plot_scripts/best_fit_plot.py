import random
import numpy as np
from functools import partial
from deap import base, tools, creator
import pickle
from os import path, sys
import matplotlib.pyplot as plt
from matplotlib import cm
from IPython import embed


## Starting the Game
## Drawing a random code
# variables to be set accoring to pickle file name
NB_COLORS=4
NB_SLOTS=100
eaLambda= 2
eaMu= 1

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



## PLOTS
## Extracting records
## Description: Reads logs from script's arguments then plot average best fitness in function of time (generations).

# Reading Files
logs = []
for pickle_file in sys.argv[1:]:
    pickled_log = open(pickle_file, 'rb')
    logs.append(pickle.load(pickled_log))
# Extracting data from logs
gen = []
bestFit = []
for log in logs:
    log_gen = log.select('generation')
    gen = log_gen if len(log_gen) > len(gen) else gen
    log_min = [ x for x in log.select('min') if x is not None]
    bestFit.append(log_min)

# Building array of average best fitnesses
bestFit = [l+[0]*(len(gen)-len(l)) for l in bestFit]
bestFit = np.mean(np.array(bestFit), axis=0)

## Min Fitness against running time
fig, ax = plt.subplots()
ax.plot(gen, bestFit, 'b*', label='set your label')
ax.set_xlabel("Generation")
ax.set_ylabel("Best Fitness")
#plt.ylim(NB_SLOTS, 0)
plt.legend()
plt.show()

