import random
import numpy as np
from functools import partial
from deap import base, tools, creator
#from IPython import embed
import pickle
from os import path, mkdir
import argparse


## Script's argument parser
### Matstermind game
parser = argparse.ArgumentParser(description="Build your Genetic Algorithm to solve the Mastermind game.")
parser.add_argument('-n', required=True, dest='NB_SLOTS', type=int, action='store', metavar='nb_slots', help='Size of the Mastermind game (number of holes/slots).')
parser.add_argument('-k', required=True, dest='NB_COLORS', type=int, action='store', metavar='nb_colors', help='Number of colors in the Mastermind game.')

### Genetic Algorithm
parser.add_argument('-l', required=True, dest='lambda_', type=int, metavar='lambda', help='Number of offsprings of a given population in the (m[+/,]lambda)-GA.')
parser.add_argument('-m', required=True, dest='mu', type=int, metavar='mu', help='Size of intitial and post-selection populations in the (m[+/,]lambda)-GA.')
parser.add_argument('--mll', action='store_true', help='(mu+(lambda, lambda))-Genetic Algorithm.')

### Strategy
parser.add_argument('--max-gen', dest='maxGen', type=int, metavar='Tmax', default=10000, help='Stop algorithm after Tmax generations.')
parser.add_argument('--mutpb', dest='mutpb', type=float, metavar='mutation_pb', help='Probability that an offspring color sequence is a result of mutation.')
parser.add_argument('--xpb', dest='xpb', type=float, default=0.7, metavar='crossover_pb', help='Probability that an offspring color sequence is a result of a crossover.')
parser.add_argument('--selection', dest='sel', choices=['comma', 'plus'], default='plus', help='Selection strategy: "comma" for (mu,lambda)-GA and "plus" for (mu+lambda)-GA')
parser.add_argument('--var-ps', dest='ps', choices=['fp', 'random', 'elitist'], default='fp', help='Parent selection strategy in the variation phase: random, elitist or fitness proportional')

#### Operators
parser.add_argument('--op-mutpb', dest='ind_mutpb', metavar='ind_mutpb', default=0.3, help='Independent probability for a color to be mutated. Use "fp" for fitness proportional probability.')
parser.add_argument('--op-xpb', dest='ind_xpb', type=float, default=0.5, metavar='ind_xpb', help='Independent probability for a color to be changed in a Uniform abd Blend crossover.')
parser.add_argument('--op-cx', dest='cx_op', choices=['uniform', 'blend', 'one-point', 'two-point'], default = 'uniform',  help='Crossover operator. Default is Unifrm crossover.')
# hna dir tournament: parser.add_argument('--op-selection', dest='sel_op', choices=['fp', 'elitist'], default='elitist', help='Selection strategy: either elitist or fitness proportional.')
parser.add_argument('--fp-selection', dest='fp_sel', action='store_true', help='Selection strategy: either elitist or fitness proportional.')

#### Logging, statistics
parser.add_argument('--log', dest='log', choices=['record', 'save'], help='Enable stastics recording. Stored in a DEAP logbook. Record then inspect using your favorite interactive interpreter, otherwise use "save" to write the logs into a pickled file.')

#### Progress bar
parser.add_argument('--show-progress', dest='prog_bar', choices=['true', 'false'], default='true', help='Display progress bar. This requires tqdm to be installed.')


## Parsing / errot checking
args = parser.parse_args()
if args.sel == 'comma' and args.lambda_ < args.mu:
    parser.error('Parameter lambda must be greater than parameter mu in Comma selection.')
if args.ind_mutpb != 'fp' and  args.ind_mutpb != None:
    try:
        args.ind_mutpb = float(args.ind_mutpb)
    except:
        parser.error('ind_mutpb should be set to either "fp" or a valid probability value.')
if args.mutpb != None and args.xpb + args.mutpb > 1:
    parser.error('The sum of mutation_pb and crossover_pb must be inferior to 1.')
if args.prog_bar == 'true':
    from tqdm import trange



## Starting the Game
## Drawing a random code
NB_COLORS=args.NB_COLORS
NB_SLOTS=args.NB_SLOTS
masterCode = np.random.choice(NB_COLORS, NB_SLOTS)
print("Master code:", masterCode)
print("")

## Evaluating guesses against the master code
def evalGuess(guess, code=masterCode):
    if len(guess) != NB_SLOTS:
        raise Exception("evalGuess expects guess of length %d."%NB_SLOTS)
    for color in guess:
        if color not in range(NB_COLORS):
            raise Exception("Unknown color %s."%color)

    diff = code - np.array(guess)
    return len(diff[diff!=0])


eaLambda= args.lambda_
eaMu= args.mu


# Container of individual guesses
creator.create('distanceMin', base.Fitness, weights=(-1.0,))
creator.create('Guess', np.ndarray, fitness=creator.distanceMin)
## Preparing the Evoltionary Algorithm Operators
## EA toolbox
eaBox = base.Toolbox()
## Random guesses generator
eaBox.register('guess', tools.initIterate, creator.Guess, partial(np.random.choice, NB_COLORS, NB_SLOTS ))
eaBox.register('population', tools.initRepeat, list, eaBox.guess)

# Variation Operators:
## Sample with 
# Mutation operators:
## Independent mutations of guess entries
def mutUniformFP(guess):
    return tools.mutUniformInt(guess, low=0, up=NB_COLORS-1, indpb=guess.fitness.values[0]/NB_SLOTS)
if args.ind_mutpb=='fp':
    eaBox.register('mutate', mutUniformFP)
else:
    eaBox.register('mutate', tools.mutUniformInt, low=0, up=NB_COLORS-1, indpb=args.ind_mutpb)


# Crossover operators:
## Uniform crossover over entries (independent for each entry with proba indpb)
if args.cx_op == 'one-point':
    eaBox.register('cross', tools.cxOnePoint)
elif args.cx_op == 'two-point':
    eaBox.register('cross', tools.cxTwoPoint)
elif args.cx_op == 'blend':
    eaBox.register('cross', tools.cxBlend, alpha=0)
else:
    eaBox.register('cross', tools.cxUniform, indpb=args.ind_xpb)


# Selection operators:
## Select / Sample with fitness.proportional probability:
def selFitProbable(k, population, fitnesses):
    fitnesses = np.array(fitnesses)+0.001
    try:
        pbs = fitnesses/fitnesses.sum()
    except (RuntimeError):
        pass
    selected_idxs = np.random.choice(len(population), size=k, replace=False, p=pbs)
    if k==1:
        return (population[selected_idxs[0]],)
    else:
        return [population[idx] for idx in selected_idxs]

eaBox.register('selectFP', selFitProbable)
eaBox.register('select', tools.selTournament, k=eaMu, tournsize=2, fit_attr='fitness')
eaBox.register('selectBest', tools.selBest, k=eaMu,  fit_attr='fitness')
eaBox.register('selectBestOne', tools.selBest, k=1, fit_attr='fitness')


# Logs & statistics
if args.log != None:
    stats = tools.Statistics(key=lambda guess: guess.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    log = tools.Logbook()

# Evolution Strategy
def mu2xLambda(population, lambda_=eaLambda, psel=None):
    # Population of the next generation
    nextGenPop = []
    # Lambda Mutations
    for _ in range(0, lambda_):
        g, = eaBox.mutate(eaBox.clone(population[0]))
        del g.fitness.values
        g.fitness.values = (evalGuess(g, masterCode),)
        nextGenPop.append(g)
    if psel=='elitist':
        best_g, = eaBox.selectBestOne(nextGenPop)
    elif psel=='fp':
        try: 
            best_g, = eaBox.selectFP(1, nextGenPop, map(lambda g: NB_SLOTS-g.fitness.values[0], nextGenPop))
        except:
            best_g, = eaBox.selectBestOne(nextGenPop)
    else:
        best_g = random.choice(nextGenPop)
    #Lambda crossovers
    for _ in range(0, lambda_):
        cg, _ = eaBox.cross(eaBox.clone(population[0]),best_g)
        del cg.fitness.values
        cg.fitness.values = (evalGuess(cg, masterCode),)
        nextGenPop.append(cg)
    return nextGenPop
 

def muLambda(population, lambda_=eaLambda, xpb=0.7, mutpb=None, psel=None):
    # Population of the next generation
    nextGenPop = []
    mutpb = 1-xpb if mutpb is None else mutpb
    # Crossover or Mutate
    for _ in range(0, lambda_):
        xOrmut = random.random()
        if xOrmut < xpb:
            if psel=='fp':
                # Fitness proportional crossover parent selection
                invFitnesses = map(lambda g: NB_SLOTS-g.fitness.values[0], population) 
                g1, g2 = map(eaBox.clone, eaBox.selectFP(2, population, invFitnesses))
            else:
                g1, g2 = random.sample(population, 2)
            g1, g2  = eaBox.cross(g1, g2)
            del g1.fitness.values
            g1.fitness.values = (evalGuess(g1, masterCode),)
            nextGenPop.append(g1)
        elif xOrmut < xpb + mutpb:
            g = eaBox.clone(random.choice(population))
            g, = eaBox.mutate(g)
            del g.fitness.values
            g.fitness.values = (evalGuess(g, masterCode),)
            nextGenPop.append(g)
        else:
            nextGenPop.append(eaBox.clone(random.choice(population)))
    return nextGenPop


#def ga_mastermind(mm_game, ga, logging=False):
def ga_mastermind():

    # Random initialization
    initPopulation = eaBox.population(eaMu)
    for guess in initPopulation:
        guess.fitness.values = (evalGuess(guess, masterCode),)
    bestFitness, = eaBox.selectBestOne(initPopulation)[0].fitness.values

    # Stopping condition
    # Max number of iteration
    maxGen = args.maxGen
    currentPopulation = initPopulation
    iterator = trange(maxGen, ncols=70) if args.prog_bar=='true' else xrange(maxGen)
    #for iteration in tqdm(xrange(maxGen), ncols=70):
    for iteration in iterator:
        if (0,)  in [guess.fitness.values for guess in currentPopulation]:
            break

        # Variation
        if args.mll:
            nextGenPopulation = mu2xLambda(currentPopulation, lambda_=args.lambda_, psel=args.ps)
        else:
            nextGenPopulation = muLambda(currentPopulation, xpb=args.xpb, mutpb=args.mutpb, psel=args.ps)
        
        # Selection
        if args.sel == 'comma':
            currentPopulation = nextGenPopulation
        else:
            currentPopulation = currentPopulation + nextGenPopulation

        if args.fp_sel:
            currentPopulation = eaBox.selectFP(eaMu, currentPopulation, map(lambda g: NB_SLOTS-g.fitness.values[0], currentPopulation))
        else:
            currentPopulation = eaBox.selectBest(currentPopulation, k=eaMu)

        if eaMu == 1:
            best = currentPopulation[0]
        else:
            best, = eaBox.selectBestOne(currentPopulation)

        if args.log != None:
            record = stats.compile(currentPopulation)
            log.record(generation=iteration, best_so_far=best, **record)


    if best.fitness.values[0] ==0:
        print("")
        print("Found the master code:", best, " after %d generation."%(iteration+1))
    else:
        print("Best guess:", best, "of fitness:", best.fitness.values[0]," after %d generation."%(iteration+1))

    if args.log == 'save':
        logs_dirpath = path.join(path.dirname(path.abspath(__file__)), 'logs') 
        if not path.exists(logs_dirpath):
            mkdir(logs_dirpath)
        
        i = 0
        while path.exists(path.join(logs_dirpath,'%d+%d_ga_n%d_k%d_t%d.pickle'%(eaMu, eaLambda, NB_COLORS, NB_SLOTS, i))):
            i+=1
        
        pickle.dump(log, open(path.join(path.dirname(path.realpath(__file__)), 'logs/%d+%d_ga_n%d_k%d_t%d.pickle'%(eaMu, eaLambda, NB_COLORS, NB_SLOTS, i)), 'wb'))

ga_mastermind()


