import csv
import pandas as pd
from stylegan.metrics import linear_separability
from collections import defaultdict
from glob import glob
from random import choice, sample


def get_data():
    train_file_path = "../content/drive/MyDrive/ExplainedKinshipData/data/train_pairs.csv"
    train_folders_path = "../data/train-faces/"
    val_families = "F09"

    all_images = glob(train_folders_path + "*/*/*.jpg")

    train_images = [x for x in all_images if val_families not in x]
    val_images = [x for x in all_images if val_families in x]

    train_person_to_images_map = defaultdict(list)

    ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]

    for x in train_images:
        train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

    val_person_to_images_map = defaultdict(list)

    for x in val_images:
        val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

    relationships = pd.read_csv(train_file_path)
    relationships = list(zip(relationships.p1.values, relationships.p2.values))
    relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]

    train = [x for x in relationships if val_families not in x[0]]
    val = [x for x in relationships if val_families in x[0]]
    df_pairs = pd.read_csv("/content/drive/MyDrive/ExplainedKinshipData/data/train-pairs.csv")
    print(df_pairs)
    features = []
    for pair in df_pairs:
        current_features_1 = linear_separability.get_features(pair["p1"], pair["ptype"])
        current_features_2 = linear_separability.get_features(pair["p2"], pair["ptype"])
        features.append([current_features_1, current_features_2])

    df_pairs.insert(-1, "features", features, True)
    print(df_pairs)
    return df_pairs
