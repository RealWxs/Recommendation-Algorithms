from CF.svdpp import SVDPP
import pickle
from loguru import logger
import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv('./ratingLogs.csv')
    train_set = df.sample(frac=0.8)
    val_set = df.drop(train_set.index)
    with open('../ml_ls/userNeighbour.pkl', 'rb') as f:
        unlt: dict = pickle.load(f)
    with open('./SVDPP.log','w') as f:
        f.write('')
    logger.add('./SVDPP.log')
    model = SVDPP(610, 9724, 42, unlt)
    model.train_val(train_set,val_set,600)
