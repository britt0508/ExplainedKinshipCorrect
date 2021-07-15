import csv
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns

# DATA_PATH = "/content/drive/MyDrive/ExplainedKinshipData/data/all_pairs_also_unrelated_complete.csv"
# data = pd.read_csv(DATA_PATH, converters={"feat1": literal_eval, "feat2": literal_eval})
# print(data.describe)
# feature_numbers = [19, 13, 16, 23, 20, 22, 2, 21, 17, 10, 12, 11, 24, 14, 15, 1, 0, 18, 29, 4, 34, 28, 6, 39, 5, 25, 30,
#                  31, 32, 35, 37, 27, 36, 26, 38, 3, 33, 8, 7, 9]
# feature_numbers2 = [40 + i for i in feature_numbers]
# all_feature_numbers = feature_numbers + feature_numbers2

# feature_names = ["feature_" + str(x) for x in feature_numbers]
# feature_names2 = ["feature_" + str(x) for x in feature_numbers2]
# print(feature_names)
# # print(data["feat1"].to_list())


# split_data = pd.DataFrame(data["feat1"].to_list(), columns=feature_names)
# split_data2 = pd.DataFrame(data["feat2"].to_list(), columns=feature_names2)

# complete_data = pd.concat([data[["pic1", "pic2", "ptype"]], split_data, split_data2], axis=1)
# print(complete_data)
# complete_data.to_csv("/content/drive/MyDrive/ExplainedKinshipData/data/split_features_data.csv")

DATA_PATH = "/content/drive/MyDrive/ExplainedKinshipData/data/split_features_data.csv"
data = pd.read_csv(DATA_PATH)
pd.set_option('display.expand_frame_repr', False)
print(data.ptype.value_counts())

for i in range(40):
    data["combined_feature_" + str(i)] = data["feature_" + str(i)] + data["feature_" + str(i+40)]

combined_features = [col for col in data if col.startswith("combined_feature_")]
correlation = data[combined_features].corr()
plt.figure(figsize=(10, 10))
sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
plt.savefig("correlation.png")
# print(data.describe())





