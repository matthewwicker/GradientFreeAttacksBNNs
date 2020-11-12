# Implementing a genetic algorithm to find adversarial examples.
import sys
from tqdm import tqdm
sys.path.append('../')
import numpy as np
rng = np.random.default_rng()

# Returns a length-n numpy array whose elements are all arr. A new object, not a reference.
def multiply(arr, n):
    return np.asarray(n*[arr[:]])

# This is the only selection operator that is vectorized.
def tournament2(f):
    N = f.shape[1]
    parents = np.zeros_like(f, dtype='int')
    for i in range(N):
        [c1, c2] = rng.choice(N, size=2)
        fighters1 = f[:, c1]
        fighters2 = f[:, c2]
        battle = fighters1 > fighters2
        parents[:, i] += c1*battle
        parents[:, i] += c2*np.invert(battle)
    return parents

# This is the only crossover operator that is vectorized.
def single2(pop, parents):  # pretty good
    L = pop.shape[0]
    N = pop.shape[1]
    new_pop = np.zeros_like(pop)
    for i in range(0, N, 2):
        moms = parents[:, i]
        dads = parents[:, i+1]
        mom_squares = pop[range(L), moms]
        dad_squares = pop[range(L), dads]
        cross_point = rng.choice(32)
        if rng.choice(2) == 0:
            new_pop[:, i] = np.append(mom_squares[:, :cross_point, :, :], dad_squares[:, cross_point:, :, :], axis=1)
            new_pop[:, i+1] = np.append(dad_squares[:, :cross_point, :, :], mom_squares[:, cross_point:, :, :], axis=1)
        else:
            new_pop[:, i] = np.append(mom_squares[:, :, :cross_point, :], dad_squares[:, :, cross_point:, :], axis=2)
            new_pop[:, i+1] = np.append(dad_squares[:, :, :cross_point, :], mom_squares[:, :, cross_point:, :], axis=2)
    return(new_pop)

# This is the only mutation operator that is vectorized.
def one_pixel(new_pop, R):  # pretty good with R = 0.5
    mutated_pop = new_pop
    N = new_pop.shape[1]
    for i in range(N):
        if rng.uniform() < R:
            x, y = rng.choice(32, size=2)
            mutated_pop[:, i, x, y] *= -1
    return mutated_pop

# Takes in a SINGLE input numpy array. For now, only an untargeted attack. Parameters are:
# G: number of generations
# R: mutation rate
# N: population size
# D: perturbation size
def cifar_gen_attack(model, inp, G, R, N, D, selection='t2', crossover='s2', mutation='1p'):

    # load in population or make a new one if this is the beginning of a new run
    try:
        pop = np.load('pop_in_progress.npy')
    except:
        pop = rng.choice([-D, D], size=(inp.shape[0], N, inp.shape[1], inp.shape[2], inp.shape[3]))

    # make copies of inputs to be used in the future
    copies = np.zeros((inp.shape[0], N, inp.shape[1], inp.shape[2], inp.shape[3]))
    for i in range(inp.shape[0]):
        for j in range(N):
            copies[i][j] += inp[i]

    # compute initial fitness
    clipped = np.asarray(np.clip(copies + pop, 0, 1))
    preds = np.asarray(model.predict(clipped.reshape((-1, inp.shape[1], inp.shape[2], inp.shape[3])), n=15)).reshape((inp.shape[0], N, -1))
    true_classes = np.argmax(model.predict(inp, n=15), axis=1)
    true_softmaxes = preds[range(inp.shape[0]), :, true_classes]
    f = -np.log(true_softmaxes)

    for g in tqdm(range(G)):
        
        # Do selection.
        if selection == 't2':
            parents = tournament2(f)
        elif selection == 'prop':
            parents = proportional(f)
        elif selection == 'soft':
            parents = soft(f)
        elif selection == 'rank':
            parents = rank(f)
        else:
            return 'invalid selection!'
        
        # Do crossover.
        if crossover == 's':
            new_pop = single(pop, parents)
        elif crossover == 'd':
            new_pop = double(pop, parents)
        elif crossover == 's2':
            new_pop = single2(pop, parents)
        elif crossover == 'r':
            new_pop = respectful(pop, parents)
        else:
            return 'invalid crossover!'

        # Do mutation.
        if mutation == 'rp':
            new_pop = rate_pixel(new_pop)
        elif mutation == '1p':
            new_pop = one_pixel(new_pop, R)
        elif mutation == '6p':
            new_pop = six_pixel(new_pop)
        elif mutation == '10p':
            new_pop = ten_pixel(new_pop)
        else:
            return 'invalid mutation!'

        # update pop and fitness
        pop = new_pop
        clipped = np.asarray(np.clip(copies + pop, 0, 1))
        preds = np.asarray(model.predict(clipped.reshape((-1, inp.shape[1], inp.shape[2], inp.shape[3])), n=15)).reshape((inp.shape[0], N, -1))
        true_softmaxes = preds[range(inp.shape[0]), :, true_classes]
        f = -np.log(true_softmaxes)

        # save pop
        np.save('pop_in_progress.npy', pop)

    # return the best members of the population
    best = np.argmax(f, axis=1)
    return np.clip(inp + pop[range(inp.shape[0]), best], 0, 1)

