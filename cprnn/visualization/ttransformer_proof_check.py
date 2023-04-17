import torch
import numpy as np


def check_orthogonal(m: np.ndarray):
    return m.shape[0] == m.shape[1] and np.allclose(np.linalg.norm(m, axis=0), np.ones(m.shape[0])) \
           and np.allclose(np.identity(m.shape[0]), m.T @ m) \
           and np.allclose(np.identity(m.shape[0]), m @ m.T)


def check_fullrank(m: np.ndarray):
    return min(m.shape[0], m.shape[1]) == np.linalg.matrix_rank(m)


if __name__ == '__main__':

    # Dims
    n = 128
    d = 128

    # Make bases

    # 1. permuted identity basis
    ident = torch.eye(n)
    perm_ident = ident[np.random.permutation(n)]

    # 2. random orthogonal basis
    rand_ortho = torch.randn(n, n)
    u, s, vt = torch.linalg.svd(rand_ortho)
    rand_ortho = u

    # 3. random basis
    fullrank = False
    rand = None
    while not fullrank:
        rand = torch.randn(n, n)
        fullrank = check_fullrank(rand.numpy())

    e_bases = {
        "permuted identity": perm_ident,
        "rand_ortho": rand_ortho,
        "random basis": rand
    }

    for name, e_basis in e_bases.items():

        print("({}) Orthogonal: {}".format(name, check_orthogonal(e_basis.numpy())))

        a_coeffs = torch.randn(n, d)
        b_coeffs = torch.randn(n, d)

        outer_softmax = torch.softmax(e_basis @ a_coeffs @ b_coeffs.T @ e_basis.T, dim=-1)
        inner_softmax = e_basis @ torch.softmax(a_coeffs @ b_coeffs.T, dim=-1) @ e_basis.T
        print("({}) inner softmax == outer softmax: {}".format(
            name, torch.allclose(outer_softmax, inner_softmax, atol=1e-01, rtol=1e-01)
        ))

    # Output
    # >> (permuted identity) Orthogonal: True
    # >> (permuted identity) inner softmax == outer softmax: True
    # >> (rand_ortho) Orthogonal: False
    # >> (rand_ortho) inner softmax == outer softmax: False
    # >> (random basis) Orthogonal: False
    # >> (random basis) inner softmax == outer softmax: False

