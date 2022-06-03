import pickle

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from loguru import logger


class MovieData(Dataset):
    def __init__(self, data):
        with open('../AE/embeded/f_embed3.pkl', 'rb') as f1, \
                open('../AE/embeded/attrs.pkl', 'rb') as f2, \
                open('../ml-100k/mfelt.pkl', 'rb') as f3, \
                open('../ml-100k/ufelt.pkl', 'rb') as f4:
            self.embed_f = pickle.load(f1)
            self.data = data
            self.embed_age = nn.Embedding(90, 3)
            self.embed_zipcode = nn.Embedding(999, 3)
            self.embed_sex = nn.Embedding(2, 3)
            self.embed_occupation = nn.Embedding(22, 3)
            self.attrs = pickle.load(f2)
            self.mfelt = pickle.load(f3)
            self.ufelt = pickle.load(f4)
        self.data = data.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        age, sex, occupation, zipcode = self.attrs[item[0]]

        t_occupation = self.embed_occupation(torch.LongTensor([occupation]))
        t_sex = self.embed_sex(torch.LongTensor([sex]))
        t_age = self.embed_age(torch.LongTensor([age]))

        t_zipcode = self.embed_zipcode(torch.LongTensor([zipcode]))

        tf = torch.Tensor(self.embed_f[item[1]])
        t = torch.cat((tf, t_occupation, t_sex, t_age, t_zipcode), dim=1)

        su = torch.Tensor(self.ufelt[item[0]])
        sm = torch.Tensor(self.mfelt[item[1]])

        s = torch.cat((su, sm))
        return t, s, item[2]


class XDeepFM(nn.Module):
    def __init__(self, shape):
        super(XDeepFM, self).__init__()
        features = 15
        fc_merge = 1 + 15 + 5
        fc_sparse = 45

        self.fc1 = nn.Linear(features, shape[0])
        self.fc2 = nn.Linear(shape[0], shape[1])
        self.fc_merge = nn.Linear(fc_merge, 1)
        self.fc_sparse = nn.Linear(fc_sparse, 1)
        self.ReLU = nn.ReLU()
        self.conv2dsl1 = []
        for i in range(3):
            self.conv2dsl1.append(nn.Conv2d(3, 5, (5, 5)))

        self.conv2dsl2 = []
        for i in range(3):
            self.conv2dsl2.append(nn.Conv2d(3, 5, (5, 5)))

        self.conv2dsl3 = []
        for i in range(3):
            self.conv2dsl3.append(nn.Conv2d(3, 5, (5, 5)))

        self.mp1d = nn.MaxPool1d(3)
        self.avg_pooling = nn.AvgPool1d(3)

    def mlp_branch(self, x: torch.Tensor):
        out = self.ReLU(self.fc1(x))
        out = self.ReLU(self.fc2(out))
        return out

    def cin_branch(self, x):
        x = x.view(5, -1)
        mat = torch.zeros((5, 5, 3))
        for i in range(3):
            mat[:, :, i] = x[:, i] @ x[:, i].T
        # 5*3 after conv2d

        res1 = []
        for con2dl1 in self.conv2dsl1:
            res1.append(torch.squeeze(con2dl1(torch.permute(mat, (2, 0, 1)))))
        cross1 = torch.stack(res1, dim=1)
        c1 = self.avg_pooling(cross1)

        mat2 = torch.zeros((5, 5, 3))

        for i in range(3):
            mat2[:, :, i] = cross1[:, i] @ x[:, i].T
        res2 = []
        for con2dl2 in self.conv2dsl2:
            res2.append(torch.squeeze(con2dl2(torch.permute(mat, (2, 0, 1)))))
        cross2 = torch.stack(res2, dim=1)
        c2 = self.avg_pooling(cross2)

        mat3 = torch.zeros((5, 5, 3))
        for i in range(3):
            mat3[:, :, i] = cross2[:, i] @ x[:, i].T
        res3 = []
        for con2dl3 in self.conv2dsl3:
            res3.append(torch.squeeze(con2dl3(torch.permute(mat, (2, 0, 1)))))
        cross3 = torch.stack(res3, dim=1)
        c3 = self.avg_pooling(cross3)

        return torch.cat((c1, c2, c3))*3

    def sparse_branch(self, x):
        out = self.fc_sparse(x)
        return out

    def forward(self, dense, sparse):
        y_sparse = self.sparse_branch(sparse)
        y_mlp = self.mlp_branch(dense).view(1, 5)
        y_cin = self.cin_branch(dense).view(1, 15)
        x_merge = torch.cat((y_sparse, y_mlp, y_cin), dim=1)
        y = self.fc_merge(x_merge)
        return y


if __name__ == '__main__':
    with open('./XDeepFM.log', 'w') as f:
        f.write('')

    patience = 20
    best = torch.inf
    logger.add('./XDeepFM.log')
    df = pd.read_csv('../ml-100k/u.data', header=None, delimiter="\t")
    df[0] = df[0].apply(lambda x: x - 1)
    df[1] = df[1].apply(lambda x: x - 1)
    train_set = df.sample(frac=0.8)
    val_set = df.drop(train_set.index)
    train_set = MovieData(train_set)
    trainLoader = DataLoader(train_set, batch_size=1)
    val_set = MovieData(val_set)
    valLoader = DataLoader(val_set, batch_size=1)
    net = XDeepFM((8, 5))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())

    for epoch in range(50):
        l = 0
        count = 0
        if patience == 0:
            break
        logger.success('start training epoch:{}'.format(epoch))
        for dense, sparse, y in trainLoader:
            optimizer.zero_grad()
            pred = net.forward(dense, sparse)
            loss = criterion(torch.squeeze(pred), torch.squeeze(y.float()))
            l += loss.data
            count += 1
            loss.backward()
            optimizer.step()

            if count % 5000 == 0:
                logger.info("MSE loss:{} for iter:{}".format(l / count, count))

        val_loss = 0
        count = 0
        for dense, sparse, y in valLoader:
            with torch.no_grad():
                pred = net.forward(dense, sparse)
                loss = (torch.squeeze(pred) - torch.squeeze(y.float())) ** 2
                val_loss += loss.data
                count += 1
        rmse_loss = np.sqrt(val_loss / count)
        logger.debug('validation set RMSE loss:{}'.format(rmse_loss))
        if rmse_loss < best:
            best = rmse_loss
            logger.success('RMSE loss {} is the new lowest one'.format(rmse_loss))
            patience = 10
        else:
            logger.success('hit patience, remain:{}, current best is {}'.format(patience, best))
            patience -= 1