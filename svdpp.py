import pickle
import random

from loguru import logger
import numpy as np
import pandas
import pandas as pd
from CF.plotutil import show


def to_uid(uid):
    return uid - 1


class SVDPP:
    def __init__(self, users, movies, latent, ublt):
        self.MU = np.random.random()
        self.BU = np.random.random([users, 1])
        self.BI = np.random.random([movies, 1])
        self.Q = np.random.random([movies, latent])
        self.P = np.random.random([users, latent])
        self.Y = np.random.random([movies, latent])
        self.ublt = ublt
        self.PFU = None
        self.latent = latent
        self.l = 0.0001
        self.lr = 0.001
        self.buffer_train = []
        self.buffer_val = []
        self.best = np.inf
        self.patience = 10

    def forward(self, u, m):
        PFU = np.zeros([1, self.latent])
        neighbour = self.ublt.get(u)

        if neighbour is not None:
            length = len(neighbour)
            for j in neighbour:
                PFU += self.Y[j]
            PFU = PFU / np.sqrt(length)

        PFU = PFU + self.P[u]
        self.PFU = PFU
        return self.MU + self.BU[u] + self.BI[m] + np.dot(PFU, self.Q[m].T)

    def J(self, g, u, m):
        penaltyJ = 0
        neighbour = self.ublt.get(u)
        if neighbour is not None:
            for j in neighbour:
                penaltyJ += np.sum(self.Y[j] ** 2)
        penalty = self.BU[u] ** 2 + self.BI[m] ** 2 + np.sum(self.Q[m] ** 2) + np.sum(self.P[u] ** 2) + penaltyJ
        loss = np.squeeze(g ** 2 + self.l * penalty)
        return loss

    def backward(self, g, u, m):
        gMU = g
        gBU_user = g + self.l * self.BU[u]
        gBI_item = g + self.l * self.BI[m]
        gQ_item = g * self.PFU + self.l * self.Q[m]
        gP_user = g * self.Q[m] + self.l * self.P[u]

        neighbour = self.ublt.get(u)
        if neighbour is not None:
            gY_prefix = g * self.Q[m] / np.sqrt(len(neighbour))
            gY_j = {}
            for j in neighbour:
                gY_j[j] = gY_prefix + self.l * self.Y[j]
            for j in neighbour:
                self.Y[j] = self.Y[j] - self.lr * gY_j[j]
        self.MU = self.MU - self.lr * gMU
        self.BU[u] = self.BU[u] - self.lr * gBU_user
        self.BI[m] = self.BI[m] - self.lr * gBI_item
        self.Q[m] = self.Q[m] - self.lr * gQ_item
        self.P[u] = self.P[u] - self.lr * gP_user

    def fit(self, data: pd.DataFrame):
        loss = 0
        count = 0
        for item in data.itertuples(index=False):
            pred = self.forward(to_uid(item[0]), item[1])
            g = pred - float(item[2])
            loss += self.J(g, to_uid(item[0]), item[1])
            count += 1
            self.backward(g, to_uid(item[0]), item[1])
            if count % 5000 == 4999:
                logger.info("RMSE loss:{} for iter:{}".format(np.sqrt(loss / count), count))
        return np.sqrt(loss / count)

    def validate(self, val: pd.DataFrame):
        loss = 0
        count = 0
        for item in val.itertuples(index=False):
            pred = self.forward(to_uid(item[0]), item[1])
            g = pred - float(item[2])
            loss += self.J(g, to_uid(item[0]), item[1])
            count += 1
        return np.sqrt(loss / count)

    def train_val(self, data: pandas.DataFrame, val, epochs):
        for epoch in range(1, epochs + 1):
            if self.patience == 0:
                break
            data = data.sample(frac=1)
            logger.debug('begin training epoch:{}'.format(epoch))
            loss = self.fit(data)
            logger.success('training RMSE loss:{}'.format(loss))
            self.buffer_train.append(loss)
            logger.info('begin validation')
            loss = self.validate(val)
            self.buffer_val.append(loss)
            logger.info('RMSE loss on validation set is {}'.format(loss))
            if loss < self.best:
                self.patience = 10
                self.best = loss
                logger.success('new Lowest RMSE loss:{} on validation set'.format(loss))
            else:
                self.patience -= 1
                logger.info('hit patience, current lowest loss is {}, remain:{}'.format(self.best,self.patience))
        show(self.buffer_train, self.buffer_val)
