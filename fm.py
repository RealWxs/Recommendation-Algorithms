import pickle

import numpy as np
from loguru import logger
import pandas as pd


def J(g):
    return g ** 2


class FM:
    def __init__(self, features, latent, uelt, melt):
        self.b = np.random.random()
        self.w = np.random.random([features, 1])
        self.V = np.random.random([features, latent])
        self.FeatureSum = None
        self.x = None
        self.lr = 0.00001
        self.features = features
        self.latent = latent
        self.embed_user = uelt
        self.embed_movie = melt
        self.buffer_train = []
        self.buffer_val = []
        self.best = np.inf
        self.patience = 10

    def forward(self, x):
        self.x = x
        feature_sum = np.zeros([1, self.latent])
        ident_cross = 0
        for i in range(self.features):
            feature_sum = feature_sum + x[i] * self.V[i]
            ident_cross += self.V[i] @ self.V[i].T * x[i] ** 2
        self.FeatureSum = feature_sum
        return self.b + self.w.T @ x + 0.5 * feature_sum @ feature_sum.T - 0.5 * ident_cross

    def backward(self, g):
        gb = g
        gw = g * self.x

        for k in range(self.features):
            gVk = g * (self.x[k] * self.FeatureSum + self.x[k] ** 2 * self.V[k])
            self.V[k] = self.V[k] - self.lr * gVk
        self.b = self.b - self.lr * gb
        self.w = self.w - self.lr * gw

    def train_epoch(self, data: pd.DataFrame):
        loss = 0
        count = 0
        for item in data.itertuples(index=False):
            u_feature = self.embed_user[item[0]]
            m_feature = self.embed_movie[item[1]]
            pred = self.forward(np.concatenate((u_feature, m_feature)).reshape(-1, 1))
            g = np.squeeze(pred - item[2])
            loss += J(g)
            count += 1
            self.backward(g)
            if count % 5000 == 4999:
                logger.info('RMSE loss:{} for {} iters'.format(np.sqrt(loss / count),count))

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
    with open('../ml-100k/uelt.pkl', 'rb') as f:
        print(pickle.load(f))

