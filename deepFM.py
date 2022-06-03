import pickle

import pandas
import numpy as np
import pandas as pd
from loguru import logger

from FM.fm import J

from FM.wideAndDeep import Relu


class DeepFM:
    def __init__(self, features, latent, hidden, embed_user, embed_movie):
        self.b = np.random.random()
        self.w = np.random.random([features, 1])
        self.V = np.random.random([features, latent])
        self.FeatureSum = None
        self.lr = 0.00002
        self.features = features
        self.latent = latent
        self.w1 = np.array(np.random.uniform(-2.445 / np.sqrt(hidden[0] + features),
                                             2.445 / np.sqrt(hidden[0] + features), (hidden[0], features)))
        self.w2 = np.array(np.random.uniform(-2.445 / np.sqrt(hidden[1] + hidden[0]),
                                             2.445 / np.sqrt(hidden[1] + hidden[0]), (hidden[1], hidden[0])))
        self.w3 = np.array(np.random.uniform(-2.445 / np.sqrt(1 + hidden[1]),
                                             2.445 / np.sqrt(1 + hidden[1]), (1, hidden[1])))
        self.b1 = np.random.random([hidden[0], 1])
        self.b2 = np.random.random([hidden[1], 1])
        self.b3 = np.random.random([1, 1])
        self.w_mlp = np.random.random([1, 1])
        self.w_fm = np.random.random([1, 1])
        self.b_merge = np.random.random([1, 1])
        self.x1 = None
        self.x2 = None
        self.x3 = None
        self.p1 = None
        self.p2 = None
        self.y_MLP = None
        self.y_FM = None
        self.embed_user = embed_user
        self.embed_movie = embed_movie
        self.patience = 10
        self.buffer_train = []
        self.buffer_val = []
        self.best = np.inf

    def fm_branch(self, x):
        feature_sum = np.zeros([1, self.latent])
        ident_cross = 0
        for i in range(self.features):
            feature_sum = feature_sum + x[i] * self.V[i]
            ident_cross += self.V[i] @ self.V[i].T * x[i] ** 2
        self.FeatureSum = feature_sum
        y_fm = self.b + self.w.T @ x + 0.5 * feature_sum @ feature_sum.T - 0.5 * ident_cross
        return y_fm

    def mlp_branch(self, x):
        self.x1 = x

        out = self.w1 @ x + self.b1
        out, self.p1 = Relu(out)
        self.x2 = out

        out = self.w2 @ out + self.b2
        out, self.p2 = Relu(out)
        self.x3 = out

        y_nn = self.w3 @ out + self.b3
        return y_nn

    def forward(self, x):
        y_mlp = self.mlp_branch(x)
        y_fm = self.fm_branch(x)
        y_hat = self.w_fm * y_fm + self.w_mlp * y_mlp + self.b_merge
        self.y_FM = y_fm
        self.y_MLP = y_mlp
        return y_hat

    def backward(self, g):
        # Merge Layer backward

        g_wfm = g * self.y_FM
        g_wmlp = g * self.y_MLP
        g_bmerge = g

        # FM backward&step
        g_fm = g * self.w_fm

        gb = g_fm
        gw = g_fm * self.x1

        for k in range(self.features):
            gVk = g_fm * (self.x1[k] * self.FeatureSum + self.x1[k] ** 2 * self.V[k])
            self.V[k] = self.V[k] - self.lr * gVk
        self.b = self.b - self.lr * gb
        self.w = self.w - self.lr * gw

        # MLP backward&step
        g_mlp = g * self.w_mlp

        gb3 = g_mlp
        gw3 = g_mlp * self.x3.T
        gb2 = self.p2.T @ self.w3.T * g_mlp
        gw2 = gb2 @ self.x2.T

        gb1 = self.p1.T @ self.w2.T @ gb2
        gw1 = gb1 @ self.x1.T

        self.w1 = self.w1 - self.lr * gw1
        self.b1 = self.b1 - self.lr * gb1

        self.w2 = self.w2 - self.lr * gw2
        self.b2 = self.b2 - self.lr * gb2

        self.w3 = self.w3 - self.lr * gw3
        self.b3 = self.b3 - self.lr * gb3

        # MergeLayer step
        self.w_fm = self.w_fm - self.lr * g_wfm
        self.w_mlp = self.w_mlp - self.lr * g_wmlp
        self.b_merge = self.b_merge - self.lr*g_bmerge

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
            if self.patience == 0:
                break
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
    with open('../ml-100k/movie_ae_embed.pkl', 'rb') as f1, open('../ml-100k/uelt.pkl', 'rb') as f2:
        embed_m = pickle.load(f1)
        embed_u = pickle.load(f2)
    model = DeepFM(8, 5, (5, 5), embed_u, embed_m)
    df = pd.read_csv('../ml-100k/u.data', header=None, delimiter='\t')
    df[0] = df[0].apply(lambda x: x - 1)
    df[1] = df[1].apply(lambda x: x - 1)
    train_set = df.sample(frac=0.8)
    val_set = df.drop(train_set.index)
    with open('./deepFM.log', 'w') as f:
        f.write('')
    logger.add('./deepFM.log')
    model.train_val(train_set, val_set, 300)
