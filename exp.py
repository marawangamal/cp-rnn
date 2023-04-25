import random

import numpy as np
import matplotlib.pyplot as plt

import math

import torch
import sklearn.decomposition
from sklearn.preprocessing import StandardScaler

from transformers import BertTokenizer, BertModel, BertForNextSentencePrediction


def simulated_annealing(initial_state, outfile='bert_simaneal_expvar.png'):
    """Peforms simulated annealing to find a solution"""
    initial_temp = 90
    final_temp = .1
    alpha = .1

    current_temp = initial_temp

    # Start by initializing the current state with the initial state
    current_state = initial_state
    solution = current_state
    fmts = ['-o', '-x', '-.', '--x', '--o', '-']
    i = 0
    flag = None
    while current_temp > final_temp:
        neighbor = get_neighbor(current_state)

        # Check if neighbor is best so far
        cost_neighbor, cum_sum_eigenvalues_neighbor = get_cost(neighbor)
        cost_current, cum_sum_eigenvalues_current = get_cost(current_state)
        cost_diff = cost_current - cost_neighbor

        # if the new solution is better, accept it
        if cost_diff > 0:
            solution = neighbor
            cum_sum_eigenvalues_current = cum_sum_eigenvalues_neighbor
            flag = "Accept"
            cost = cost_neighbor
        # if the new solution is not better, accept it with a probability of e^(-cost/temp)
        else:
            if random.uniform(0, 1) < math.exp(-cost_diff / current_temp):
                solution = neighbor
                cum_sum_eigenvalues_current = cum_sum_eigenvalues_neighbor
                flag = "Accept"
                cost = cost_neighbor
            else:
                flag = "Rejected"
                cost = cost_current

        # decrement the temperature
        current_temp -= alpha
        print("{} | Temp: {} | Cost: {} | {}".format(flag, current_temp, cost,  cum_sum_eigenvalues_current))
        plt.plot(cum_sum_eigenvalues_current, fmts[i % len(fmts)])
        i += 1

        # if i > 10:
        #     break

    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(outfile)

    return solution


def get_cost(X_train):
    """Calculates cost of the argument state for your solution."""

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    X_train = torch.from_numpy(X_train).to(device)

    u, s, vt = torch.svd(X_train, some=True)

    vals = s.cpu().numpy()
    cost = sum(vals[len(vals)//6:])
    return cost, vals


def get_neighbor(X_train):
    """Returns neighbors of the argument state for your solution."""
    m, n = X_train.shape
    X_train = np.random.permutation(X_train.reshape(-1))
    X_train = X_train.reshape(m, n)
    return X_train


def explained_variance(X_train, iters=16, outfile='expvar.png'):
    fmts = ['-o', '-x', '-.', '--x', '--o', '-']
    for i in range(iters):
        print("iter: {}".format(i))
        m, n = X_train.shape
        X_train = np.random.permutation(X_train.reshape(-1))
        X_train = X_train.reshape(m, n)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        X_train = torch.from_numpy(X_train).to(device)

        u, s, vt = torch.svd(X_train, some=True)

        vals = s.cpu().numpy()
        cost = sum(vals[len(vals)//2:])

        print("Cost: {} | {}".format(cost,  vals))
        plt.plot(vals, fmts[i % len(fmts)])

    plt.ylabel('Sigmas')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(outfile)


def findoptreshape(vecs):
    n = len(vecs)
    fac_a, fac_b = 1, n
    print("finding factors")
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            print(i, n // i)
            fac_a, fac_b = i, n // i
    return fac_a, fac_b


def getresenet50mat():
    resnet50 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)

    vecs = []

    for name, param in resnet50.named_parameters():
        print(name, param.shape)
        if list(param.shape[-2:]) == [3, 3]:
            vecs.append(param.reshape(-1, 9))

    vecs = torch.cat(vecs, dim=0).reshape(-1)
    facs_a, facs_b = findoptreshape(vecs)
    vecs = vecs.reshape(facs_a, facs_b)
    vecs = vecs.detach().numpy()

    return vecs


def getbertmat():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    import pdb; pdb.set_trace()
    # model = BertModel.from_pretrained("bert-base-uncased")
    model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")

    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)

    vecs = list()

    for name, param in model.named_parameters():
        print(name, param.shape)
        if "encoder" in name and list(param.shape) == [768, 768]:
            vecs.append(param)

    vecs = torch.cat(vecs, dim=0).reshape(-1)
    facs_a, facs_b = findoptreshape(vecs)
    vecs = vecs.reshape(facs_a, facs_b)
    vecs = vecs.detach().numpy()
    import pdb; pdb.set_trace()
    return vecs

def main():

    # vecs = getresenet50mat()
    vecs = getbertmat()

    simulated_annealing(vecs, outfile='bert_simaneal_expvar.png')
    # explained_variance(vecs, iters=16, outfile='bert_expvar.png')


if __name__ == '__main__':
    main()
