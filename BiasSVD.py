import pickle
import random
from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CF.plotutil import show

with open('../ml-100k/movieIdToMy.pkl', 'rb') as f:
    mtlt: dict = pickle.load(f)


def parseMid(movie_id):
    return mtlt[movie_id]


def parseUid(user_id):
    return user_id - 1


# BiasSVD
class BiasSVD:
    def __init__(self, m, n, k):
        self.P = np.random.random([m, k])
        self.Q = np.random.random([n, k])
        self.BU = np.random.random([m, 1])
        self.BI = np.random.random([n, 1])
        self.MU = np.random.random(1)
        self.lu = 0.0001
        self.li = 0.0001
        self.lr = 0.0005
        self.mat = None
        self.train_buffer = []
        self.val_buffer = []
        self.patience = 10
        self.best = np.inf

    def forward(self, index):
        user, item = index
        return np.dot(self.P[user], self.Q[item]) + self.BU[user] + self.BI[item] + self.MU

    def backward(self, g, index):
        user, item = index

        gMU = g
        gBU_user = g + self.lu * self.BU[user]
        gBI_item = g + self.li * self.BI[item]
        gP_user = g * self.Q[item] + self.lu * self.P[user]
        gQ_item = g * self.P[user] + self.li * self.Q[item]

        self.MU -= self.lr * gMU
        self.BU[user] -= self.lr * gBU_user
        self.BI[item] -= self.lr * gBI_item
        self.P[user] -= self.lr * gP_user
        self.Q[item] -= self.lr * gQ_item

    def J(self, g, index):
        user, item = index
        return np.squeeze(g ** 2 + self.lu * (self.BU[user] ** 2 + np.sum((self.P[user]) ** 2)) + self.li * (
                self.BI[item] ** 2 + np.sum((self.Q[item]) ** 2)))

    # @jit(forceobj=True)
    # def fit_mat(self, data: np.ndarray, epochs):
    #     rows, cols = data.shape
    #     for epoc in range(epochs):
    #         loss = 0
    #         for user in range(rows):
    #             for item in range(cols):
    #                 if data[user, item] is None or data[user, item] == -1:
    #                     continue
    #                 pred = self.forward((user, item))
    #                 g = pred - data[user, item]
    #                 loss += self.J(g, (user, item))
    #                 self.backward(g, (user, item))
    #
    #         print(loss)

    def fit(self, data):
        loss = 0
        count = 0
        for item in data.itertuples(index=False):
            pred = self.forward((parseUid(item[0]), item[1]))
            g = pred - float(item[2])
            loss += self.J(g, (parseUid(item[0]), item[1]))
            count += 1
            self.backward(g, (parseUid(item[0]), item[1]))
            if count % 5000 == 4999:
                logger.info('RMSE loss:{} for {} items'.format(np.sqrt(loss / count), count))
        return np.sqrt(loss / count)

    def predict(self, my_uid, my_mid):
        return self.mat[my_uid, my_mid]

    def getMat(self):
        self.mat = self.P @ self.Q.T + self.BU + self.BI.T + self.MU

    def validate(self, data):
        loss = 0
        count = 0
        for item in data.itertuples(index=False):
            pred = self.predict(parseUid(item[0]), item[1])
            g = pred - float(item[2])
            loss += self.J(g, (parseUid(item[0]), item[1]))
            count += 1
        return np.sqrt(loss / count)

    def train_val(self, data: pd.DataFrame, epochs, val):
        for epoch in range(epochs):
            if self.patience == 0:
                logger.success('patience ran out, stopping.')
                logger.success('current lowest loss in validation set is {}'.format(self.best))
                break
            logger.debug('start training epoch: {}'.format(epoch))
            data = data.sample(frac=1)
            loss = self.fit(data)
            self.train_buffer.append(loss)
            logger.info('finish train epoch: {}, RMSE loss:{}'.format(epoch, loss))
            self.getMat()
            logger.info('start validation')
            loss = self.validate(val)
            self.val_buffer.append(loss)
            if loss < self.best:
                self.best = loss
                self.patience = 10
                logger.success('new best validation score:{}'.format(loss))
            else:
                self.patience -= 1
                logger.debug('hit patience, remain:{}'.format(self.patience))

        show(self.train_buffer, self.val_buffer)

    # def get_mat(self):
    #     return self.P @ self.Q.T + self.BU + self.BI.T + self.MU

    def get_params(self):
        params = {'P': self.P, 'Q': self.Q, 'BU': self.BU, 'BI': self.BI, 'MU': self.MU}
        return params


if __name__ == '__main__':
    # mat = np.array([[1, 2, 3],
    #                 [None, 3, None],
    #                 [3, 4, 5]])
    # mat = np.random.randint(1, 5, [20, 20])
    # y = mat[0,0]
    # mat[0, 0] = -1
    # model = Model2(20, 20, 20)
    # model.fit(mat, 500)
    # print('predicted:{},true:{}'.format(round(model.get_mat()[0,0]),y))
    # print(mat)
    # print(model.get_mat())
    # print(p)
    # print(q)

    train_set = pd.read_csv('../ml-100k/u1.base', header=None, delimiter='\t')
    val_set = pd.read_csv('../ml-100k/u1.test', header=None, delimiter='\t')
    del train_set[3]
    del val_set[3]

    # train_set[0] = train_set[0].apply(parseUid)
    # val_set[0] = val_set[0].apply(parseUid)
    train_set[1] = train_set[1].apply(parseMid)
    val_set[1] = val_set[1].apply(parseMid)

    with open('./BiasSVD.log', 'w') as f:
        f.write('')
    logger.add('./BiasSVD.log')
    model = BiasSVD(943, 1682, 42)
    model.train_val(train_set, 300, val_set)

    model.getMat()
    mat: np.ndarray = model.mat
    with open('./mat.pkl','wb') as f:
        pickle.dump(mat,f)
