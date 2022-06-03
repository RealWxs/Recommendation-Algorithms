import pickle

import numpy as np

from CF.BiasSVD import BiasSVD
from loguru import logger
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('./ratingLogs.csv')
    train_set = df.sample(frac=0.8)
    val_set = df.drop(train_set.index)
    model = BiasSVD(610, 9724, 42)

    logger.add('./BSVD_ml_ls.log')
    model.train_val(train_set, 300, val_set)
    model.getMat()
    mat: np.ndarray = model.mat
    with open('./mat.pkl', 'wb') as f:
        pickle.dump(mat, f)

