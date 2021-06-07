import numpy as np
from sklearn.linear_model import LinearRegression
import csv


DATA_PATH = "/content/drive/MyDrive/ExplainedKinshipData/data/pic_train_pairs.csv"
with open(DATA_PATH) as csv:
    x = "input x"
    y = "input y"


def LinRegression():
    model = LinRegression().fit(x, y)
    r_sq = model.score(x, y)
    y_pred = model.predict(x)
