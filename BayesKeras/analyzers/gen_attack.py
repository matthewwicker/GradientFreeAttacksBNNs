# Implementing a genetic algorithm to find adversarial examples.
import sys
from tqdm import tqdm
sys.path.append('../')
import numpy as np
rng = np.random.default_rng()

# Returns a length-n numpy array whose elements are all arr. A new object, not a reference.
def multiply(arr, n):
    return np.asarray(n*[arr[:]])

# Softmax that doesn't overflow.
def robust_softmax(x):
    z = x - np.max(x)
    return np.exp(z) / np.sum(np.exp(z))

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
        mom_squares = pop[range(L), moms].reshape((L, 28, 28))
        dad_squares = pop[range(L), dads].reshape((L, 28, 28))
        cross_point = rng.choice(28)
        if rng.choice(2) == 0:
            new_pop[:, i] = np.append(mom_squares[:, :cross_point, :], dad_squares[:, cross_point:, :], axis=1).reshape((L, 784))
            new_pop[:, i+1] = np.append(dad_squares[:, :cross_point, :], mom_squares[:, cross_point:, :], axis=1).reshape((L, 784))
        else:
            new_pop[:, i] = np.append(mom_squares[:, :, :cross_point], dad_squares[:, :, cross_point:], axis=2).reshape((L, 784))
            new_pop[:, i+1] = np.append(dad_squares[:, :, :cross_point], mom_squares[:, :, cross_point:], axis=2).reshape((L, 784))
    return(new_pop)

# This is the only mutation operator that is vectorized.
def one_pixel(new_pop, R):  # pretty good with R = 0.5
    mutated_pop = new_pop
    N = new_pop.shape[1]
    for i in range(N):
        if rng.uniform() < R:
            pix = rng.choice(784)
            mutated_pop[:, i, pix] *= -1
    return mutated_pop

# Takes in a SINGLE input numpy array. For now, only an untargeted attack. Parameters are:
# G: number of generations
# R: mutation rate
# N: population size
# D: perturbation size
def gen_attack(model, inp, G, R, N, D, selection='t2', crossover='s2', mutation='1p'):

    # SELECTION METHODS (each returns an array of N indices)

    def soft(f):  # pretty bad
        probs = robust_softmax(f)
        return rng.choice(N, p=probs, size=N)

    def proportional(f):  # pretty bad
        probs = f / np.sum(f)
        return rng.choice(N, p=probs, size=N)

    def rank(f):  # pretty good
        ranks = np.zeros(N)
        order =  np.argsort(f)
        for i in range(N):
            ranks[order[i]] = i+1
        probs = ranks / np.sum(ranks)
        return rng.choice(N, p=probs, size=N)

    # CROSSOVER METHODS (each returns an array of N perturbations)

    def single(pop, parents):  # pretty good
        new_pop = np.zeros_like(pop)
        for i in range(0, N, 2):
            mom = parents[i]
            dad = parents[i+1]
            cross_point = rng.choice(inp.shape[0])
            new_pop[i] = np.append(pop[mom][:cross_point], pop[dad][cross_point:])
            new_pop[i+1] = np.append(pop[dad][:cross_point], pop[mom][cross_point:])
        return(new_pop)

    def double(pop, parents):  # pretty good
        new_pop = np.zeros_like(pop)
        for i in range(0, N, 2):
            mom = parents[i]
            dad = parents[i+1]
            [c1, c2] = rng.choice(inp.shape[0], size=2)
            if c1 > c2:
                t = c1
                c1 = c2
                c2 = t
            new_pop[i] = np.concatenate((pop[mom][:c1], pop[dad][c1:c2], pop[mom][c2:]))
            new_pop[i+1] = np.concatenate((pop[dad][:c1], pop[mom][c1:c2], pop[dad][c2:]))
        return(new_pop)

    def respectful(pop, parents):  # terrible
        new_pop = np.zeros_like(pop)
        for i in range(0, N, 2):
            mom = parents[i]
            dad = parents[i+1]
            for j in range(784):
                if pop[mom][j] == pop[dad][j]:
                    new_pop[i][j] = pop[mom][j]
                    new_pop[i+1][j] = pop[mom][j]
                else:
                    x = rng.choice([-D, D])
                    new_pop[i][j] = x
                    new_pop[i+1][j] = -x
        return new_pop

    # MUTATION METHODS

    def rate_pixel(new_pop):  # pretty good with R = 0.0001 or 0.001
        return new_pop * rng.choice([-1, 1], p=[R, 1-R], size=new_pop.shape)

    def six_pixel(new_pop):  # slightly less good than one_pixel
        mutated_pop = new_pop
        for i in range(N):
            if rng.uniform() < R:
                pix = rng.choice(778)
                mutated_pop[i][pix:pix+6] *= -1  # we mutate a strip of contiguous pixels
        return mutated_pop

    def ten_pixel(new_pop):  # about the same as one_pixel, with smaller R
        mutated_pop = new_pop
        for i in range(N):
            if rng.uniform() < R:
                pix = rng.choice(784, size=10)
                for p in pix:
                    mutated_pop[i][p] *= -1
        return mutated_pop

    true_classes = np.argmax(model.predict(inp, n=15))
    pop = rng.choice([-D, D], size=(inp.shape[0], N, inp.shape[1]))
    # pop = rng.uniform(-D, D, size=(N, inp.shape[0]))
    copies = np.zeros((inp.shape[0], N, inp.shape[1]))
    for i in range(inp.shape[0]):
        for j in range(N):
            copies[i][j] += inp[i]
    stagnation = 0
    best_fitness_so_far = -1000

    for g in tqdm(range(G)):
        
        # Compute fitness.
        clipped = np.asarray(np.clip(copies + pop, 0, 1))
        preds = np.asarray(model.predict(clipped.reshape((-1, inp.shape[1])), n=15)).reshape((inp.shape[0], N, -1))
        true_softmaxes = preds[:, :, true_classes]
        f = -np.log(true_softmaxes)

        # Return the fittest member if it misclassifies.
        best = np.argmax(f, axis=1)
        # if np.argmax(preds[best]) != true_class:
            # print('noice!')
            # return inp + pop[best]

        # print(f[best])

        # Update stagnation and best fitness so far.
        # if f[best] > best_fitness_so_far:
            # best_fitness_so_far = f[best]
            # stagnation = 0
        # else:
            # stagnation += 1
        
        # Stop early if we're stagnated.
        # if stagnation == 50:
            # return np.clip(inp + pop[best], 0, 1)
        
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

        pop = new_pop

    return np.clip(inp + pop[range(inp.shape[0]), best], 0, 1)

