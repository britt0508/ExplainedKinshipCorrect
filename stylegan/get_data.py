import tensorflow as tf

import pandas as pd
import os
import random
import csv
import glob
import argparse
import numpy as np

import numpy as np
import sklearn.svm
import pickle
import cv2

from PIL import Image
from pathlib import Path
from collections import defaultdict

import stylegan.dnnlib as dnnlib
import stylegan.dnnlib.tflib as tflib

from stylegan import config

import sys

# tf.compat.v1.enable_eager_execution()


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


def open_file_or_url(file_or_url):
    if dnnlib.util.is_url(file_or_url):
        return dnnlib.util.open_url(file_or_url, cache_dir=config.cache_dir)
    return open(file_or_url, 'rb')


def load_pkl(file_or_url):
    with open_file_or_url(file_or_url) as file:
        dnnlib.tflib.init_tf()
        return pickle.load(file, encoding='latin1')


# Get classifier for features once
def get_features(imgs, predictions, x):
    all_images = []
    for image_input in imgs:
        image = cv2.imread(str(image_input))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image - np.mean(image, (0, 1))
        image = image / np.std(image, axis=(0, 1))
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        image = np.array(image)
        image = np.expand_dims(image.transpose((2, 0, 1)), axis=0)
        all_images.append(image)
    concat_imgs = np.concatenate(all_images)

    result = tflib.run(predictions, feed_dict={x: concat_imgs})[0]

    return result[:, 1].tolist()


def write_features_csv(args):
    # tflib.init_tf()
    train_adjective = 'test'
    BASE_PATH = args.data_dir
    # LABELS_PATH = os.path.join(BASE_PATH, "train-pairs.csv")
    IMAGE_PATH = os.path.join(BASE_PATH, "{}-faces/".format(train_adjective))
    # DATA_PATH = os.path.join(BASE_PATH, 'pic_train_pairs.csv')
    # LOADER_PATH = os.path.join(BASE_PATH, 'loader_pic_train_pairs.csv')
    FEAT_PATH = os.path.join(args.out_dir, "features_all_{}_{}.csv".format(train_adjective, str(args.feature)))
    FEAT_PATH_TEST = os.path.join(args.out_dir, "features_all_{}_{}.csv".format(train_adjective, str(args.feature)))

    # Get list of all image paths
    im_path = []
    im_path_rel = []
    for img in sorted(glob.glob(IMAGE_PATH + "*.jpg")):
        im_path.append(img)
        im_path_rel.append(os.path.relpath(img, IMAGE_PATH))
    # im_path = im_path # remove for full dataset
    print("image paths loaded")

    # Get feature values per feature for each picture:
    # [[feat1], [feat2], ..., [feat40]] with [feat1] a list of feature 1 values for all pictures
    batch_size = args.batch_size
    x = tf.placeholder(tf.float32, shape=(None, 3, 256, 256))

    url = classifier_urls[args.feature]
    c = load_pkl(url)
    logits = c.get_output_for(x, None, is_validation=True, randomize_noise=False, structure='fixed')
    predictions = [tf.nn.softmax(tf.concat([logits, -logits], axis=1))]

    row = []

    print("Performing feature: " + str(args.feature))
    im_len = len(im_path)
    for i in range(0, im_len, batch_size):
        print("Performing batch: " + str(int(i / batch_size)))
        # tf.reset_default_graph()
        # classifier = load_pkl(url)
        if im_len > i + batch_size:
            imgs = im_path[i:i + batch_size]
        else:
            imgs = im_path[i:im_len]
        feature = get_features(imgs, predictions, x)
        row.extend(feature)

        if int(i / batch_size) % 20 == 0:
            df = pd.DataFrame(np.array([im_path_rel[:len(row)]] + [row]).T,
                              columns=["Image_path"] + [args.feature])
            df.to_csv(FEAT_PATH_TEST)
            df = None

    print("classifier loaded: " + str(url))
    df = pd.DataFrame(np.array([im_path_rel[:len(row)]] + [row]).T, columns=["Image_path"] + [args.feature])
    df.to_csv(FEAT_PATH_TEST)
    return df


def write_features_pairs_csv(args):
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

                        row = [pic_path1, pic_path2, pair[0], pair[1], pair[2], features_1, features_2]
                        pic_train_writer.writerow(row)
            print(pic_train_writer)
            pic_csv.close()


def write_features_pairs_csv_test(all_features):
    BASE_PATH = "/content/drive/MyDrive/ExplainedKinshipData/data/"
    LABELS_PATH = os.path.join(BASE_PATH, "test_competition.xlsx")
    IMAGE_PATH = os.path.join(BASE_PATH, "test-faces/")

    # with open(LABELS_PATH) as csv_file:
    df_pd = pd.read_excel(LABELS_PATH, names=["index", "first", "second", "relation"])
    df_pd["features_1"] = np.nan
    df_pd["features_2"] = np.nan
    pd.set_option("display.max_columns", None)
    print(df_pd)

    for index, pair in df_pd.iterrows():
        pic_path1 = pair["first"]
        pic_1_entire_path = str(IMAGE_PATH) + str(pic_path1)
        features_1 = all_features.loc[all_features['Image_path'] == pic_path1].values.tolist()[0][3:]

        pic_path2 = pair["second"]
        pic_2_entire_path = str(IMAGE_PATH) + str(pic_path2)
        features_2 = all_features.loc[all_features['Image_path'] == pic_path2].values.tolist()[0][3:]

        names1 = {pair["index"]: features_1}
        names2 = {pair["index"]: features_2}

        df_pd['features_1'] = df_pd[pair["index"]].map(names1)
        df_pd['features_2'] = df_pd[pair["index"]].map(names2)


def get_random_pairs_nonfam():
    print("help")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# parser = argparse.ArgumentParser(description='Get a single feature for the entire dataset.')
# parser.add_argument('--feature', type=int, help='The feature we extract in this run.', default=0)
# parser.add_argument('--batch-size', type=int, help='Batch size used.', default=32)
# parser.add_argument('--data-dir', type=str, help="Directory of the dataset.", required=True)
# parser.add_argument('--out-dir', type=str, help="Output Directory", required=True)
# parser.add_argument('--train', type=str2bool, help='Boolean whether to take training or validation images.',
#                     default=True)

# args = parser.parse_args()
# write_features_csv(args)
# write_features_pairs_csv()

# path = "/content/drive/MyDrive/ExplainedKinshipData/data/Features_test/merged.csv"
# with open(path) as csv_file:
#     data = pd.read_csv(csv_file)
#     write_features_pairs_csv_test(data)

# path = "/content/drive/MyDrive/ExplainedKinshipData/data/Features_test"
# all_files = glob.glob(os.path.join(path, "features_all_test_*.csv"))
# combined_csv_data = pd.read_csv(all_files[0], delimiter=',')

# for f in all_files[1:]:
#     df = pd.read_csv(f, delimiter=',')
#     df.set_index('Image_path', inplace=True)
#     df = df.iloc[: , 1:]
#     print(df)
#     combined_csv_data = pd.merge(combined_csv_data, df, on='Image_path')

# combined_csv_data.to_csv(os.path.join(path, "merged.csv"))
# print(combined_csv_data)

