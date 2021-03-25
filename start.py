import pickle

import sklearn as sklearn
import torch
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from pathlib import Path

# model = VGG16(weights='imagenet', include_top=False)
#
# img_path = Path('C:/Users/Gebruiker/PycharmProjects/ExplainedKinship')
# img = image.load_img(img_path / 'test.jpg.jpg', target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
#
# features = model.predict(x)
# print(features.shape)
# print(features)
#000000004C6BC47E
#fsqbQVOR65947^rxvBYT}+!

# Generate uncurated MetFaces images with truncation (Fig.12 upper left)
# python generate.py --outdir=out --trunc=0.7 --seeds=600-605 --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

# with open('ffhq.pkl', 'rb') as f:
#     G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
# z = torch.randn([1, G.z_dim]).cuda()    # latent codes
# c = None                                # class labels (not used in this example)
# img = G(z, c)                           # NCHW, float32, dynamic range [-1, +1]
# w = G.mapping(z, c, truncation_psi=0.5, truncation_cutoff=8)
# img = G.synthesis(w, noise_mode='const', force_fp32=True)

