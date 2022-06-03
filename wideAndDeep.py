import pandas as pd

import numpy as np
import pickle

from loguru import logger

from FM.fm import J


def Relu(x: np.ndarray):
    p = np.diag((x >= 0).squeeze())
    if len(p.shape) < 2:
        p = 1
    return (abs(x) + x) / 2, p


class WideAndDeep:
    def __init__(self, features, hidden, cross_lookup, embed_user, embed_movie):
        self.buffer_train = []
        self.buffer_val = []
        self.patience = 20
        self.best = np.inf
        self.lr = 0.000001
        self.w = np.random.random([features + len(cross_lookup), 1])
        self.b = np.random.random([1, 1])
        self.ww = np.random.random([1, 1])
        self.wd = np.random.random([1, 1])
        self.b_merge = np.random.random([1, 1])
        self.w1 = np.array(np.random.uniform(-2.445 / np.sqrt(hidden[0] + features),
                                             2.445 / np.sqrt(hidden[0] + features), (hidden[0], features)))
        self.w2 = np.array(np.random.uniform(-2.445 / np.sqrt(hidden[1] + hidden[0]),
                                             2.445 / np.sqrt(hidden[1] + hidden[0]), (hidden[1], hidden[0])))
        self.w3 = np.array(np.random.uniform(-2.445 / np.sqrt(1 + hidden[1]),
                                             2.445 / np.sqrt(1 + hidden[1]), (1, hidden[1])))
        self.b1 = np.random.random([hidden[0], 1])
        self.b2 = np.random.random([hidden[1], 1])
        self.b3 = np.random.random([1, 1])
        self.features = features
        self.clt = cross_lookup
        self.embed_user = embed_user
        self.embed_movie = embed_movie
        self.x1 = None
        self.x2 = None
        self.x3 = None
        self.p1 = None
        self.p2 = None
        self.z = None
        self.y_w = None
        self.y_d = None

    def deep_branch(self, x):
        self.x1 = x

        out = self.w1 @ x + self.b1
        out, self.p1 = Relu(out)
        self.x2 = out

        out = self.w2 @ out + self.b2
        out, self.p2 = Relu(out)
        self.x3 = out

        y_nn = self.w3 @ out + self.b3
        return y_nn

    def wide_branch(self, x):
        cross_features = []
        for cross_index in self.clt:
            cross_features.append(x[cross_index[0]] * [cross_index[1]])
        z = np.concatenate((x, np.array(cross_features).reshape(-1, 1)))
        self.z = z
        return self.w.T @ z + self.b

    def forward(self, x):
        yw = self.wide_branch(x)
        yd = self.deep_branch(x)
        self.y_w = yw
        self.y_d = yd
        return self.ww * yw + self.wd * yd + self.b_merge

    def backward(self, g):
        # Merge Layer Backward
        g_ww = self.y_w * g
        g_wd = self.y_d * g
        g_bmerge = g

        # Wide Component Backward&Step
        g_wide = g * self.ww
        g_b = g_wide
        g_w = g_wide * self.z

        self.b = self.b - self.lr * g_b
        self.w = self.w - self.lr * g_w

        # Deep Component Backward%Step
        g_deep = g * self.wd

        gb3 = g_deep
        gw3 = g_deep * self.x3.T
        gb2 = self.p2.T @ self.w3.T * g_deep
        gw2 = gb2 @ self.x2.T

        gb1 = self.p1.T @ self.w2.T @ gb2
        gw1 = gb1 @ self.x1.T

        self.w1 = self.w1 - self.lr * gw1
        self.b1 = self.b1 - self.lr * gb1

        self.w2 = self.w2 - self.lr * gw2
        self.b2 = self.b2 - self.lr * gb2

        self.w3 = self.w3 - self.lr * gw3
        self.b3 = self.b3 - self.lr * gb3

        # Merge Layer Step
        self.ww = self.ww - self.lr * g_ww
        self.wd = self.wd - self.lr * g_wd
        self.b_merge = self.b_merge - self.lr * g_bmerge

    def train_epoch(self, data: pd.DataFrame):
        loss = 0
        count = 0
        for item in data.itertuples(index=False):
            u_feature = np.array(self.embed_user[item[0]])
            m_feature = np.array(self.embed_movie[item[1]])
            x = np.concatenate((u_feature, m_feature)).reshape(-1, 1)

            pred = self.forward(x)
            g = np.squeeze(pred - item[2])
            loss += J(g)
            count += 1
            self.backward(g)
            if count % 5000 == 4999:
                logger.info('RMSE loss:{} for {} iters'.format(np.sqrt(loss / count), count))

        return np.sqrt(loss / count)

    def validate(self, val):
        loss = 0
        count = 0
        for item in val.itertuples(index=False):
            u_feature = self.embed_user[item[0]]
            m_feature = self.embed_movie[item[1]]
            pred = self.forward(np.concatenate((u_feature, m_feature)).reshape(-1, 1))
            g = np.squeeze(pred - item[2])
            loss += J(g)
            count += 1
        return np.sqrt(loss / count)

    def train_val(self, data: pd.DataFrame, val, epochs):
        for epoch in range(epochs):
            logger.success('start training epoch:{}'.format(epoch))
            data = data.sample(frac=1)
            loss = self.train_epoch(data)
            self.buffer_train.append(loss)
            logger.success('training epoch{} done, RMSE loss:{}'.format(epoch, loss))
            logger.success('start validating')
            loss = self.validate(val)
            self.buffer_val.append(loss)
            logger.debug('RMSE loss on validation set is {}'.format(loss))
            if loss < self.best:
                self.patience = 10
                self.best = loss
                logger.success('the loss of {} is new best score'.format(loss))
            else:
                self.patience -= 1
                logger.success('hit patience, remain:{}, current best is {}'.format(self.patience, self.best))


if __name__ == '__main__':
    with open('../ml-100k/mfelt.pkl', 'rb') as f1, open('../ml-100k/ufelt.pkl', 'rb') as f2:
        embed_m = pickle.load(f1)
        embed_u = pickle.load(f2)

    cross_lookupt = []
    for u in range(26):
        for m in range(26, 26 + 19):
            cross_lookupt.append((u, m))
    for u1 in [0, 1, 2]:
        cross_lookupt.append((u1, 3))
        cross_lookupt.append((u1, 4))
        for u2 in range(5, 26):
            cross_lookupt.append((u1, u2))
            cross_lookupt.append((3, u2))
            cross_lookupt.append((4, u2))

    model = WideAndDeep(26 + 19, (5, 5), cross_lookupt, embed_u, embed_m)
    df = pd.read_csv('../ml-100k/u.data', header=None, delimiter='\t')
    df[0] = df[0].apply(lambda x: x - 1)
    df[1] = df[1].apply(lambda x: x - 1)

    train_set = df.sample(frac=0.8)
    val_set = df.drop(train_set.index)
    with open('./WideAndDeep.log', 'w') as f:
        f.write('')
    logger.add('./WideAndDeep.log')
    model.train_val(train_set, val_set, 300)
