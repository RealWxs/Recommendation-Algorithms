import pickle

import pandas as pd
import numpy as np
from loguru import logger

from FM.wideAndDeep import Relu


def softmax(x):
    x_translate = x - np.max(x)
    temp = np.exp(x_translate)
    return temp / np.sum(temp)


def J(y_hat, y):
    delta = 1e-9
    res = -np.sum(y * np.log(y_hat + delta))
    return res


def RMSE(y_hat, y):
    return (np.argmax(y_hat) - np.argmax(y)) ** 2


class DCN:
    def __init__(self, features, hidden, embed_user, embed_movie, classes):
        self.buffer_train = []
        self.buffer_val = []
        self.patience = 20
        self.best = np.inf
        self.features = features
        self.embed_user = embed_user
        self.embed_movie = embed_movie
        self.lr = 1e-4
        self.classes = classes
        self.w1 = np.array(np.random.uniform(-2.445 / np.sqrt(hidden[0] + features),
                                             2.445 / np.sqrt(hidden[0] + features), (hidden[0], features)))
        self.w2 = np.array(np.random.uniform(-2.445 / np.sqrt(hidden[1] + hidden[0]),
                                             2.445 / np.sqrt(hidden[1] + hidden[0]), (hidden[1], hidden[0])))
        self.w3 = np.array(np.random.uniform(-2.445 / np.sqrt(hidden[2] + hidden[1]),
                                             2.445 / np.sqrt(hidden[2] + hidden[1]), (hidden[2], hidden[1])))
        self.b1 = np.random.random([hidden[0], 1])
        self.b2 = np.random.random([hidden[1], 1])
        self.b3 = np.random.random([hidden[2], 1])
        self.x1 = None
        self.x2 = None
        self.x3 = None
        self.p1 = None
        self.p2 = None
        self.cx2 = None
        self.cx3 = None
        self.cdotw2 = None
        self.cdotw3 = None
        self.y_cross = None
        self.y_deep = None
        self.y_merge = None
        self.cw1 = np.random.random([features, 1])
        self.cw2 = np.random.random([features, 1])
        self.cw3 = np.random.random([features, 1])
        self.cb1 = np.random.random([features, 1])
        self.cb2 = np.random.random([features, 1])
        self.cb3 = np.random.random([features, 1])

        self.wm = np.random.random([classes, features + hidden[2]])
        self.bm = np.random.random([classes, 1])

    def branch_deep(self, x):
        self.x1 = x

        out = self.w1 @ x + self.b1
        out, self.p1 = Relu(out)
        self.x2 = out

        out = self.w2 @ out + self.b2
        out, self.p2 = Relu(out)
        self.x3 = out

        y_deep = self.w3 @ out + self.b3
        return y_deep

    def branch_cross(self, cx1):
        cx2 = cx1 * (cx1.T @ self.cw1) + self.cb1 + cx1
        self.cx2 = cx2
        self.cdotw2 = cx1.T @ self.cw2
        cx3 = cx2 * self.cdotw2 + self.cb2 + cx2
        self.cx3 = cx3
        self.cdotw3 = cx1.T @ self.cw3
        y_cross = cx3 * self.cdotw3 + self.cb3 + cx3
        return y_cross

    def forward(self, x):
        y_cross = self.branch_cross(x)
        y_deep = self.branch_deep(x)
        self.y_cross = y_cross
        self.y_deep = y_deep
        self.y_merge = np.concatenate((y_cross, y_deep))
        return softmax(self.wm @ self.y_merge + self.bm)

    def backward(self, g):
        # Merge Layer Backward
        gbm = g
        gwm = g @ self.y_merge.T

        # Cross Component Backward&Step
        g_cross = g.T @ self.wm[:, :self.features]
        g_crossT = g_cross.T
        gcw3 = self.x1 * (self.cx3.T @ g_crossT)
        gcb3 = g_crossT

        gcw2 = (1 + self.cdotw3) * self.x1 * (self.cx2.T @ g_crossT)
        gcb2 = (1 + self.cdotw3) * g_crossT

        gcw1 = (1 + self.cdotw3) * (1 + self.cdotw2) * self.x1 * (self.x1.T @ g_crossT)
        gcb1 = (1 + self.cdotw3) * (1 + self.cdotw2) * g_crossT

        self.cw3 = self.cw3 - self.lr * gcw3
        self.cb3 = self.cb3 - self.lr * gcb3

        self.cw2 = self.cw2 - self.lr * gcw2
        self.cb2 = self.cb2 - self.lr * gcb2

        self.cw1 = self.cw1 - self.lr * gcw1
        self.cb1 = self.cb1 - self.lr * gcb1

        # Deep Component Backward&Step
        g_deep = (self.wm[:, self.features:]).T @ g

        gb3 = g_deep
        gw3 = g_deep @ self.x3.T
        gb2 = self.p2.T @ self.w3.T @ g_deep
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
        self.wm = self.wm - self.lr * gwm
        self.bm = self.bm - self.lr * gbm

    def train_epoch(self, data: pd.DataFrame):
        loss = 0
        count = 0
        for item in data.itertuples(index=False):
            u_feature = np.array(self.embed_user[item[0]])
            m_feature = np.array(self.embed_movie[item[1]])
            x = np.concatenate((u_feature, m_feature)).reshape(-1, 1)
            pred = self.forward(x)

            y = np.zeros([self.classes, 1])
            y[item[2] - 1] = 1
            g = pred - y
            loss += J(pred, y)
            count += 1
            self.backward(g)
            if count % 5000 == 4999:
                logger.info('CE loss:{} for {} iters'.format(loss / count, count))

        return loss / count

    def validate(self, val):
        loss = 0
        count = 0
        correct = 0
        for item in val.itertuples(index=False):
            u_feature = self.embed_user[item[0]]
            m_feature = self.embed_movie[item[1]]
            pred = self.forward(np.concatenate((u_feature, m_feature)).reshape(-1, 1))
            y = np.zeros([self.classes, 1])
            if np.argmax(pred) == item[2] - 1:
                correct += 1
            y[item[2] - 1] = 1
            loss += J(pred, y)
            count += 1
        return loss / count, correct / count

    def train_val(self, data: pd.DataFrame, val, epochs):
        for epoch in range(epochs):
            logger.success('start training epoch:{}'.format(epoch))
            data = data.sample(frac=1)
            loss = self.train_epoch(data)
            self.buffer_train.append(loss)
            logger.success('training epoch{} done, CE loss:{}'.format(epoch, loss))
            logger.success('start validating')
            loss, acc = self.validate(val)
            self.buffer_val.append(loss)
            logger.debug('CE loss on validation set is {},accuracy is {}'.format(loss, acc))
            if loss < self.best:
                self.patience = 10
                self.best = loss
                logger.success('the loss of {} is new best score'.format(loss))
            else:
                self.patience -= 1
                logger.success('hit patience, remain:{}, current best is {}'.format(self.patience, self.best))


if __name__ == '__main__':
    # with open('../ml-100k/movie_ae_embed.pkl', 'rb') as f1, open('../ml-100k/uelt.pkl', 'rb') as f2:
    #     embed_m = pickle.load(f1)
    #     embed_u = pickle.load(f2)
    # model = DCN(8, (5, 5, 3), embed_u, embed_m, 5)
    # df = pd.read_csv('../ml-100k/u.data', header=None, delimiter='\t')
    # df[0] = df[0].apply(lambda x: x - 1)
    # df[1] = df[1].apply(lambda x: x - 1)
    # train_set = df.sample(frac=0.8)
    # val_set = df.drop(train_set.index)
    # with open('./DCN.log', 'w') as f:
    #     f.write('')
    # logger.add('./DCN.log')
    # model.train_val(train_set, val_set, 300)

    df = pd.read_csv('../ml-100k/u.data', header=None, delimiter='\t')
    df[0] = df[0].apply(lambda x: x - 1)
    df[1] = df[1].apply(lambda x: x - 1)


