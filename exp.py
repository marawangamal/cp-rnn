import numpy as np
import matplotlib.pyplot as plt


import math

import torch
import sklearn.decomposition
from sklearn.preprocessing import StandardScaler


def explained_variance(X_train, iters=2):

    fmts = ['-o', '--']
    for i in range(iters):
        print("iter: {}".format(i))
        X_train = np.random.permutation(X_train)
        X_trainT = np.random.permutation(X_train.T)
        X_train = X_trainT.T

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

        # Create the visualization plot

        # plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
        # plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',
        #          label='Cumulative explained variance')

        plt.plot(exp_var_pca, fmts[i])
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
    for i in range(1, int(math.sqrt(n))+1):
        if n % i == 0:
            print(i, n//i)
            fac_a, fac_b = i, n//i

    vecs = vecs.reshape(fac_a, fac_b)
    print("vecs shape: {}".format(vecs.shape))

    vecs = vecs.detach().numpy()
    explained_variance(vecs)

if __name__ == '__main__':
    main()