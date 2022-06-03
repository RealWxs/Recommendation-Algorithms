import numpy as np
from numpy.linalg import inv
from numba import jit


def softmax(x: np.ndarray):
    m = max(x)
    temp = np.exp(x - m)
    return temp / sum(temp)


def get_trans_prob(mat: np.ndarray):
    users, items = mat.shape
    M = np.zeros([users + items, users + items])
    for u in range(users):
        edges = np.sum(mat[u])
        M[u, users:] = M[u, users:] + (mat[u, :] / edges)
    for i in range(items):
        edges = np.sum(mat[:, i])
        M[users + i, :users] += mat[:, i] / edges
    return M


def get_weighted_trans_prob(mat: np.ndarray):
    mat[mat == 0] = 2.5
    mat = mat * 2
    users, items = mat.shape
    M = np.zeros([users + items, users + items])
    for u in range(users):
        M[u, users:] = M[u, users:] + softmax(mat[u, :])
    for i in range(items):
        M[users + i, :users] += softmax(mat[:, i])
    return M


@jit(forceobj=True)
def personal_rank(trans_prob: np.ndarray, users, items, user, alpha, epochs):
    a = users + items
    r = np.zeros([a, 1])
    r[user] = 1
    last_r = r
    for epoch in range(epochs):
        r = (1 - alpha) * inv(np.identity(a) - alpha * trans_prob.T) @ r
        diff = np.sum((r - last_r) ** 2)

        print("epoch:{}, diff:{}".format(epoch, diff))
        if diff < 1e-7:
            break
        last_r = r
    return r


if __name__ == '__main__':
    # m = np.zeros([4, 5])
    # m[0, 0] = m[0, 2] = m[1, 1] = m[1, 3] = m[2, 0] = m[2, 1] = m[2, 4] = m[3, 0] = m[3, 1] = m[3, 2] = 1
    #
    # trans_prob_w = get_weighted_trans_prob(m)
    # trans_prob = get_trans_prob(m)
    #
    # res1 = (personal_rank(trans_prob, 4, 5, 0, 0.8, 20)[:4])
    # res2 = (personal_rank(trans_prob_w, 4, 5, 0, 0.8, 20)[:4])
    # print(res1)
    # print(res2)
    a = np.array([1, 2, 3])
