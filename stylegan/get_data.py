from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import torch
import torchvision
from PIL import Image
import torch
import os
import random
import csv
import glob
import tensorflow as tf
from stylegan.metrics import linear_separability

BASE_PATH = "/content/drive/MyDrive/ExplainedKinshipData/data/"
LABELS_PATH = BASE_PATH + "train-pairs.csv"
IMAGE_PATH = BASE_PATH + "train-faces/"
DATA_PATH = BASE_PATH + 'pic_train_pairs.csv'
LOADER_PATH = BASE_PATH + 'loader_pic_train_pairs.csv'
FEAT_PATH = BASE_PATH + "features_all"


def write_features_csv():
    with open(FEAT_PATH, mode='w') as feat_csv:
        feat_csv = csv.writer(feat_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        feat_csv.writerow(["pic_path", "features"])

        for img in glob.glob(IMAGE_PATH + "***/**/*.jpg"):
            print("IMAGE HERE PLEASE")
            features = linear_separability.get_features(img)
            row = [img, features]
            feat_csv.writerow(row)

def write_features_pairs_csv():
    with open(LABELS_PATH) as csv_file:
        pairs_data = csv.reader(csv_file)
        next(pairs_data, None)

        all_features = pd.read_csv(FEAT_PATH)

        with open(str(BASE_PATH) + 'pic_train_pairs.csv', mode='w') as pic_csv:
            pic_train_writer = csv.writer(pic_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            pic_train_writer.writerow(["pic1", "pic2", "p1", "p2", "ptype"])

            for pair in pairs_data:
                print(pair)
                for picture_pair1 in os.listdir(str(IMAGE_PATH) + str(pair[0])):
                    for picture_pair2 in os.listdir(str(IMAGE_PATH) + str(pair[1])):
                        # Look at each picture separately
                        pic_path1 = str(pair[0]) + "/" + str(picture_pair1)
                        pic_path2 = str(pair[1]) + "/" + str(picture_pair2)
                        pic_1_entire_path = str(IMAGE_PATH) + str(pic_path1)
                        pic_2_entire_path = str(IMAGE_PATH) + str(pic_path2)

                        # Get features from stylegan lin sep per picture
                        features_1 = all_features.loc[df['pic_path'] == pic_1_entire_path]
                        features_2 = all_features.loc[df['pic_path'] == pic_2_entire_path]

                        # features_1 = linear_separability.get_features(str(IMAGE_PATH) + str(pic_path1))
                        # features_2 = linear_separability.get_features(str(IMAGE_PATH) + str(pic_path2))

                        # print(features_1)
                        row = [pic_path1, pic_path2, pair[0], pair[1], pair[2], features_1, features_2]
                        pic_train_writer.writerow(row)
            print(pic_train_writer)
            pic_csv.close()


write_features_csv()
write_features_pairs_csv()




