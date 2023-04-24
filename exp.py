import random

import numpy as np
import matplotlib.pyplot as plt

import math

import torch
import sklearn.decomposition
from sklearn.preprocessing import StandardScaler


def simulated_annealing(initial_state):
    """Peforms simulated annealing to find a solution"""
    initial_temp = 90
    final_temp = .1
    alpha = .01

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
    plt.savefig('expvar.png')

    return solution


def get_cost(X_train):
    """Calculates cost of the argument state for your solution."""

    # Scale the dataset; This is very important before you apply PCA
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    # Instantiate PCA
    pca = sklearn.decomposition.PCA()

    # Determine transformed features
    X_train_pca = pca.fit_transform(X_train_std)

    # Determine explained variance using explained_variance_ration_ attribute
    exp_var_pca = pca.explained_variance_ratio_

    # Cumulative sum of eigenvalues; This will be used to create step plot
    # for visualizing the variance explained by each principal component.

    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    return sum(cum_sum_eigenvalues[:5]), cum_sum_eigenvalues


def get_neighbor(X_train):
    """Returns neighbors of the argument state for your solution."""
    m, n = X_train.shape
    X_train = np.random.permutation(X_train.reshape(-1))
    X_train = X_train.reshape(m, n)
    return X_train

def explained_variance(X_train, iters=16):
    fmts = ['-o', '-x', '-.', '--x', '--o', '-']
    for i in range(iters):
        print("iter: {}".format(i))
        m, n = X_train.shape
        X_train = np.random.permutation(X_train.reshape(-1))
        X_train = X_train.reshape(m, n)

        # Scale the dataset; This is very important before you apply PCA
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        # Instantiate PCA
        pca = sklearn.decomposition.PCA()

        # Determine transformed features
        X_train_pca = pca.fit_transform(X_train_std)

        # Determine explained variance using explained_variance_ration_ attribute
        exp_var_pca = pca.explained_variance_ratio_

        # Cumulative sum of eigenvalues; This will be used to create step plot
        # for visualizing the variance explained by each principal component.

        cum_sum_eigenvalues = np.cumsum(exp_var_pca)
        print(cum_sum_eigenvalues)

        # Create the visualization plot

        # plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
        # plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',
        #          label='Cumulative explained variance')

        plt.plot(cum_sum_eigenvalues, fmts[i % len(fmts)])
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('expvar.png')


def main():
    resnet50 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)

    vecs = []

    for name, param in resnet50.named_parameters():
        print(name, param.shape)
        if list(param.shape[-2:]) == [3, 3]:
            vecs.append(param.reshape(-1, 9))

    vecs = torch.cat(vecs, dim=0).reshape(-1)
    n = len(vecs)

    # find factors
    fac_a, fac_b = 1, n
    print("finding factors")
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            print(i, n // i)
            fac_a, fac_b = i, n // i

    vecs = vecs.reshape(fac_a, fac_b)
    print("vecs shape: {}".format(vecs.shape))

    vecs = vecs.detach().numpy()
    simulated_annealing(vecs)
    # explained_variance(vecs)


if __name__ == '__main__':
    main()
