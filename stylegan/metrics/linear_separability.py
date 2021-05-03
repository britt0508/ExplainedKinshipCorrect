# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Linear Separability (LS)."""

from collections import defaultdict
import numpy as np
import sklearn.svm
import tensorflow as tf
import stylegan.dnnlib.tflib as tflib
import stylegan.dnnlib as dnnlib
import pickle
import cv2.cv2 as cv2

from PIL import Image
from stylegan import config
from stylegan.metrics import metric_base
from stylegan.training import misc

import sys
sys.path.append("/content/ExplainedKinshipCorrect/stylegan/")


BASE_PATH = "C:/Users/Gebruiker/PycharmProjects/ExplainedKinship"
#----------------------------------------------------------------------------

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


def prob_normalize(p):
    p = np.asarray(p).astype(np.float32)
    assert len(p.shape) == 2
    return p / np.sum(p)


def mutual_information(p):
    p = prob_normalize(p)
    px = np.sum(p, axis=1)
    py = np.sum(p, axis=0)
    result = 0.0
    for x in range(p.shape[0]):
        p_x = px[x]
        for y in range(p.shape[1]):
            p_xy = p[x][y]
            p_y = py[y]
            if p_xy > 0.0:
                result += p_xy * np.log2(p_xy / (p_x * p_y)) # get bits as output
    return result


def entropy(p):
    p = prob_normalize(p)
    result = 0.0
    for x in range(p.shape[0]):
        for y in range(p.shape[1]):
            p_xy = p[x][y]
            if p_xy > 0.0:
                result -= p_xy * np.log2(p_xy)
    return result


def conditional_entropy(p):
    # H(Y|X) where X corresponds to axis 0, Y to axis 1
    # i.e., How many bits of additional information are needed to where we are on axis 1 if we know where we are on axis 0?
    p = prob_normalize(p)
    y = np.sum(p, axis=0, keepdims=True) # marginalize to calculate H(Y)
    return max(0.0, entropy(y) - mutual_information(p)) # can slip just below 0 due to FP inaccuracies, clean those up.


class LS(metric_base.MetricBase):
    def __init__(self, num_samples, num_keep, attrib_indices, minibatch_per_gpu, **kwargs):
        assert num_keep <= num_samples
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.num_keep = num_keep
        self.attrib_indices = attrib_indices
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, Gs, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu

        # Construct TensorFlow graph for each GPU.
        result_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()

                # Generate images.
                latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                dlatents = Gs_clone.components.mapping.get_output_for(latents, None, is_validation=True)
                images = Gs_clone.components.synthesis.get_output_for(dlatents, is_validation=True, randomize_noise=True)

                # Downsample to 256x256. The attribute classifiers were built for 256x256.
                if images.shape[2] > 256:
                    factor = images.shape[2] // 256
                    images = tf.reshape(images, [-1, images.shape[1], images.shape[2] // factor, factor, images.shape[3] // factor, factor])
                    images = tf.reduce_mean(images, axis=[3, 5])

                # Run classifier for each attribute.
                result_dict = dict(latents=latents, dlatents=dlatents[:, -1])
                for attrib_idx in self.attrib_indices:
                    classifier = misc.load_pkl(classifier_urls[attrib_idx])
                    logits = classifier.get_output_for(images, None)
                    predictions = tf.nn.softmax(tf.concat([logits, -logits], axis=1))
                    result_dict[attrib_idx] = predictions
                result_expr.append(result_dict)

        # Sampling loop.
        results = []
        for _ in range(0, self.num_samples, minibatch_size):
            results += tflib.run(result_expr)
        results = {key: np.concatenate([value[key] for value in results], axis=0) for key in results[0].keys()}

        # Calculate conditional entropy for each attribute.
        conditional_entropies = defaultdict(list)
        for attrib_idx in self.attrib_indices:
            # Prune the least confident samples.
            pruned_indices = list(range(self.num_samples))
            pruned_indices = sorted(pruned_indices, key=lambda i: -np.max(results[attrib_idx][i]))
            pruned_indices = pruned_indices[:self.num_keep]

            # Fit SVM to the remaining samples.
            svm_targets = np.argmax(results[attrib_idx][pruned_indices], axis=1)
            for space in ['latents', 'dlatents']:
                svm_inputs = results[space][pruned_indices]
                try:
                    svm = sklearn.svm.LinearSVC()
                    svm.fit(svm_inputs, svm_targets)
                    svm.score(svm_inputs, svm_targets)
                    svm_outputs = svm.predict(svm_inputs)
                except:
                    svm_outputs = svm_targets # assume perfect prediction

                # Calculate conditional entropy.
                p = [[np.mean([case == (row, col) for case in zip(svm_outputs, svm_targets)]) for col in (0, 1)] for row in (0, 1)]
                conditional_entropies[space].append(conditional_entropy(p))

        # Calculate separability scores.
        scores = {key: 2**np.sum(values) for key, values in conditional_entropies.items()}
        self._report_result(scores['latents'], suffix='_z')
        self._report_result(scores['dlatents'], suffix='_w')


def open_file_or_url(file_or_url):
    if dnnlib.util.is_url(file_or_url):
        return dnnlib.util.open_url(file_or_url, cache_dir=config.cache_dir)
    return open(file_or_url, 'rb')


def load_pkl(file_or_url):
    with open_file_or_url(file_or_url) as file:
        dnnlib.tflib.init_tf()
        return pickle.load(file, encoding='latin1')


def get_features(p, ptype):
    tflib.init_tf()
    # classifier = pickle.load(classifier_urls[0])
    # result_dict = dict()
    no_urls = len(classifier_urls)
    # image_path = BASE_PATH + "/train-faces/" + p
    # image_small = cv2.imread("/content/drive/MyDrive/ExplainedKinshipData/data/train-faces/F0001/MID1/P00001_face0.jpg")
    image = cv2.imread("/content/example.jpg")
    norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    print(norm_image.shape)
    # if image.shape[2] > 256:
    #     factor = image.shape[2] // 256
    #     image = tf.reshape(image, [-1, image.shape[1], image.shape[2] // factor, factor, image.shape[3] // factor, factor])
    #     image = tf.reduce_mean(image, axis=[3, 5])
    #     print(image.shape)

    # top, bottom, left, right = 66, 66, 74, 74

    # image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    image = cv2.resize(norm_image, (256, 256))
    image = np.array(image)
    print(image.shape)
    image = np.expand_dims(norm_image.T, axis=0)
    print(image.shape)
    # cv2.imshow('image small', image_small)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    results = []

    for i in range(no_urls):
        classifier = load_pkl(classifier_urls[i])
        # print(tf.rank(image, name=None))


        # Dimension 1 in both shapes must be equal, but are 256 and 3. Shapes are [1,256,256,3] and [?,3,256,256].
        # image = tf.transpose(image, [0, 2, 3, 1])
        # print(image.shape)

        # The Conv2D op currently only supports the NHWC tensor format on the CPU. The op was given the format: NCHW
        # data_format='NCHW'

        logits = classifier.get_output_for(image, None, is_validation=True, randomize_noise=True)
        # image = tflib.convert_images_to_uint8(logits)
        # end_image = image.get_output_for(image)
        print(logits)
        print("Yay")
        predictions = tf.nn.softmax(tf.concat([logits, -logits], axis=1))

        y = tf.placeholder(tf.int64, shape=(None, 10))
        acc_bool = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        acc_Num = tf.cast(acc_bool, tf.float32)
        acc_Mean = tf.reduce_mean(acc_Num)

        print(acc_bool, acc_Mean, acc_Num)

        result_dict = [predictions]
        # pred = []
        # pred.append(result_dict)
        # print(predictions.op)

        # predicted_indices = tf.argmax(predictions, 1)
        # print(predicted_indices)
        # predicted_class = tf.gather(TARGET_LABELS, predicted_indices)

        # Sampling loop.
        # results = []
        # for _ in range(0, self.num_samples, minibatch_size):
        #     results += tflib.run(result_expr)
        # results = {key: np.concatenate([value[key] for value in results], axis=0) for key in results[0].keys()}

        results += tflib.run(result_dict)[0].tolist()
        # print(results)
        # results = {key: np.concatenate([value[key] for value in results], axis=0) for key in results[0].keys()}
    print(results)

    # Shapes:
    # Tensor("celebahq-classifier-04-young_1/images_in:0", shape=(256, 256, 3), dtype=uint8)
    # Tensor("celebahq-classifier-04-young_1/labels_in:0", shape=(256, 0), dtype=float32)


    # print(classifier)


    # latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
    # images = Gs_clone.get_output_for(latents, None, is_validation=True, randomize_noise=True)
    # images = tflib.convert_images_to_uint8(images)
    # result_expr.append(inception_clone.get_output_for(images))

get_features("p", "fs")
