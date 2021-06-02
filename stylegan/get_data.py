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

classifier_urls = [
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-00-male.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-01-smiling.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-02-attractive.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-03-wavy-hair.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-04-young.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-05-5-o-clock-shadow.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-06-arched-eyebrows.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-07-bags-under-eyes.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-08-bald.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-09-bangs.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-10-big-lips.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-11-big-nose.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-12-black-hair.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-13-blond-hair.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-14-blurry.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-15-brown-hair.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-16-bushy-eyebrows.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-17-chubby.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-18-double-chin.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-19-eyeglasses.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-20-goatee.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-21-gray-hair.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-22-heavy-makeup.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-23-high-cheekbones.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-24-mouth-slightly-open.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-25-mustache.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-26-narrow-eyes.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-27-no-beard.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-28-oval-face.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-29-pale-skin.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-30-pointy-nose.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-31-receding-hairline.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-32-rosy-cheeks.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-33-sideburns.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-34-straight-hair.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-35-wearing-earrings.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-36-wearing-hat.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-37-wearing-lipstick.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-38-wearing-necklace.pkl',
    'https://nvlabs-fi-cdn.nvidia.com/stylegan/networks/metrics/celebahq-classifier-39-wearing-necktie.pkl',
]
no_urls = len(classifier_urls)


def write_features_csv():
    print("just get me those features man")
    # with open(FEAT_PATH, mode='w') as feat_csv:
    #     feat_csv = csv.writer(feat_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     feat_csv.writerow(["pic_path", "features"])

    # Get list of all image paths
    im_path = []
    for img in glob.glob(IMAGE_PATH + "***/**/*.jpg"):
        im_path.append(img)
    print("image paths loaded")

    # Get feature values per feature for each picture:
    # [[feat1], [feat2], ..., [feat40]] with [feat1] a list of feature 1 values for all pictures
    features = []
    count2 = 0
    for url in classifier_urls:
        classifier = linear_separability.load_pkl(url)
        count = 0
        row = []
        batch_size = 128
        im_len = len(im_path)
        for i in range(0, im_len, batch_size):
            if im_len > i+batch_size:
                imgs = im_path[i:i+batch_size]
            else:
                imgs = im_path[i:im_len]
            feature = linear_separability.get_features(imgs, classifier)
            row.append(feature)
            count += 1
            print(count)
        count2 += 1
        print(count2)
        print(feature)

        features.append(row)
        print("classifier loaded: " + str(url))
    df = pd.DataFrame(im_path + features, column = ["Image_path"] + [str(i) for i in no_urls])
    df.to_csv(FEAT_PATH)
    return df


def write_features_pairs_csv():
    with open(LABELS_PATH) as csv_file:
        pairs_data = csv.reader(csv_file)
        next(pairs_data, None)

        all_features = pd.read_csv(FEAT_PATH)
        # all_features = write_features_csv()


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
                        features_1 = all_features.loc[all_features['Image_path'] == pic_1_entire_path]
                        features_2 = all_features.loc[all_features['Image_path'] == pic_2_entire_path]

                        # features_1 = linear_separability.get_features(str(IMAGE_PATH) + str(pic_path1))
                        # features_2 = linear_separability.get_features(str(IMAGE_PATH) + str(pic_path2))

                        # print(features_1)
                        row = [pic_path1, pic_path2, pair[0], pair[1], pair[2], features_1, features_2]
                        pic_train_writer.writerow(row)
            print(pic_train_writer)
            pic_csv.close()


write_features_csv()
# write_features_pairs_csv()




