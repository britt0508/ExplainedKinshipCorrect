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


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

class FIW_Train(data.Dataset):
    """Dataset class for FIW Training Set"""

    def __init__(self, root_dir, labels_path, n_classes, transform):
        self.root_dir = root_dir
        self.labels_path = labels_path
        self.n_classes = n_classes
        self.transform = transform
        self.train_dataset = []
        self.preprocess()

    def preprocess(self):
        """Process the labels file"""
        lines = [line.rstrip() for line in open(self.labels_path, 'r')]
        print(lines)
        firstLine = lines.pop(0)
        for l in lines:
            spt = l.split(",")
            fname = spt[0]
            label = int(spt[5])
            # label_vec = [1 if i == label else 0 for i in range(self.n_classes)]

            self.train_dataset.append([fname, label])

    def __getitem__(self, index):
        """Return an image"""
        filename, label = self.train_dataset[index]
        image = Image.open(os.path.join(self.root_dir, filename))
        return self.transform(image), label

    def __len__(self):
        """Return the number of images."""
        return len(self.train_dataset)


class FIW_Val(data.Dataset):
    """Dataset class for FIW Validation Set"""

    def __init__(self, base_dir, csv_path, transform):
        self.base_dir = base_dir
        self.csv_path = csv_path
        self.transform = transform
        self.val_dataset = []
        self.preprocess()

    def preprocess(self):
        """Process the pair CSVs"""
        with open(self.csv_path, 'r') as f:
            re = csv.reader(f)
            lines = list(re)

        self.val_dataset = [(l[2], l[3], bool(int(l[1]))) for l in lines]

    def __getitem__(self, index):
        """Return a pair"""
        path_a, path_b, label = self.val_dataset[index]
        img_a = self.transform(Image.open(os.path.join(self.base_dir, path_a)))
        img_b = self.transform(Image.open(os.path.join(self.base_dir, path_b)))

        return (img_a, img_b), label

    def __len__(self):
        """Return the number of images."""

        return len(self.val_dataset)


def get_train_loader(image_dir, labels_path, n_classes=300, image_size=(112, 96), batch_size=16, num_workers=1):
    """Build and return a data loader for the training set."""
    transform = []

    # Only used in training
    transform.append(T.RandomHorizontalFlip())

    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = FIW_Train(image_dir, labels_path, n_classes, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader


def get_val_loader(base_dir, csv_path, image_size=(112, 96), batch_size=128, num_workers=1):
    """Build and return a data loader for a split in the validation set."""
    transform = []
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    dataset = FIW_Val(base_dir, csv_path, transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)
    return data_loader


BASE_PATH = "/content/drive/MyDrive/ExplainedKinshipData/data/"
LABELS_PATH = BASE_PATH + "train-pairs.csv"
IMAGE_PATH = BASE_PATH + "train-faces/"
DATA_PATH = BASE_PATH + 'pic_train_pairs.csv'
LOADER_PATH = BASE_PATH + 'loader_pic_train_pairs.csv'
FEAT_PATH = BASE_PATH + "features_all"

with open(FEAT_PATH, mode='w') as feat_csv:
    feat_csv = csv.writer(feat_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    feat_csv.writerow(["pic_path", "features"])

    for img in glob.glob(IMAGE_PATH + "***/**/*.jpg"):
        print("IMAGE HERE PLEASE")
        print(img)
        features = linear_separability.get_features(img)
        row = [img, features]
        feat_csv.writerow(row)


# with open(LABELS_PATH) as csv_file:
#     pairs_data = csv.reader(csv_file)
#     next(pairs_data, None)

#     all_features = pd.read_csv(FEAT_PATH)

#         with open(str(BASE_PATH) + 'pic_train_pairs.csv', mode='w') as pic_csv:
#             pic_train_writer = csv.writer(pic_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#             pic_train_writer.writerow(["pic1", "pic2", "p1", "p2", "ptype"])

#             for pair in pairs_data:
#                 print(pair)
#                 for picture_pair1 in os.listdir(str(IMAGE_PATH) + str(pair[0])):
#                     for picture_pair2 in os.listdir(str(IMAGE_PATH) + str(pair[1])):
#                         # Look at each picture separately
#                         pic_path1 = str(pair[0]) + "/" + str(picture_pair1)
#                         pic_path2 = str(pair[1]) + "/" + str(picture_pair2)
#                         pic_1_entire_path = str(IMAGE_PATH) + str(pic_path1)
#                         pic_2_entire_path = str(IMAGE_PATH) + str(pic_path2)

#                         # Get features from stylegan lin sep per picture
#                         features_1 = all_features.loc[df['pic_path'] == pic_1_entire_path]
#                         features_2 = all_features.loc[df['pic_path'] == pic_2_entire_path]

#                         # features_1 = linear_separability.get_features(str(IMAGE_PATH) + str(pic_path1))
#                         # features_2 = linear_separability.get_features(str(IMAGE_PATH) + str(pic_path2))

#                         # print(features_1)
#                         row = [pic_path1, pic_path2, pair[0], pair[1], pair[2], features_1, features_2]
#                         pic_train_writer.writerow(row)
#             print(pic_train_writer)
#             pic_csv.close()


def train_loader():
    transform = T.Compose([T.ToTensor(),
                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root=LOADER_PATH, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                         download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                           shuffle=False, num_workers=2)

    classes = ("bb", "ss", "sibs", "md", "ms", "fd",
               "fs", "gmgd", "gmgs", "gfgd", "gfgs")

    return trainloader

    # get some random training images
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    # print(images, labels)

    # training_set = get_train_loader(image_dir, labels_path)
    # print(training_set)


def baseline_model(loader):
    dataiter = iter(loader)
    images, labels = dataiter.next()
    print(images, labels)
    for i, data in enumerate(loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        print(inputs)
        print(labels)

        # Perform stylegan linear separability on the pictures
        # features_1 = linear_separability.get_features(inputs)
        # features_2 = linear_separability.get_features(inputs)

# loader = train_loader()
# baseline_model(loader)




