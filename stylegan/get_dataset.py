import csv
import pandas as pd
from stylegan.metrics import linear_separability


def get_data():
    df_pairs = pd.read_csv("C:/Users/Gebruiker/PycharmProjects/ExplainedKinship/train-pairs.csv")
    print(df_pairs)
    features = []
    for pair in df_pairs:
        current_features_1 = linear_separability.get_features(pair["p1"], pair["ptype"])
        current_features_2 = linear_separability.get_features(pair["p2"], pair["ptype"])
        features.append([current_features_1, current_features_2])

    df_pairs.insert(-1, "features", features, True)
    return df_pairs
