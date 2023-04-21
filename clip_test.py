#这个文件是测试clip对prompt和sentiment的敏感度的

import os
import clip
import torch
from torchvision.datasets import CIFAR100
import PIL
from PIL import Image
import json
import numpy as np
from torch import nn

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('RN101', device)
def read_np(path):
    content = np.load(path, allow_pickle=True, encoding='ASCII')
    if isinstance(content, np.ndarray):
        return { "features": content }

    keys = content.keys()
    if len(keys) == 1:
        return { "features": content[list(keys)[0]] }
    return content

# Download the dataset
# cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs
# image, class_id = cifar100[3637]
image = Image.open("/home/wyf/open_source_dataset/paul-gauguin_clearing-1873.jpg")
content = read_np(
    "/home/wyf/open_source_dataset/artemis_dataset/wikiart_CLIP_101_49/camille-pissarro_poultry-market-pontoise-1892.npz")
global_feat = content['g_feature']
image_input = preprocess(image).unsqueeze(0).to(device)
pool = json.load(open("/home/wyf/open_source_dataset/artemis_dataset/4.15/clean_dict_i.json", 'r'))
# classes = pool["Impressionism"]['0'] + ['people'] + ['person']
classes = ['cat']
sent = "It's a painting of "
text_inputs = torch.cat([clip.tokenize(f"{sent}{c}") for c in classes]).to('cuda')
emo_table = ["awesome", "content", "excited", "angry", "disgusting", "fearful", "sad"]

# x = clip.tokenize('amusement').to('cuda')
# y = torch.cat([clip.tokenize(f"{c}") for c in emo_table]).to('cuda')
# with torch.no_grad():
#     x_feat = model.encode_text(x)
#     y_feat = model.encode_text(y)
# x_feat /= x_feat.norm(dim=-1, keepdim=True)
# y_feat /= y_feat.norm(dim=-1, keepdim=True)
# sim1 = (100.0 * x_feat @ y_feat.T).softmax(dim=-1)
# values, indices = sim1[0].topk(7)
# for value, index in zip(values, indices):
#     print(f"{emo_table[index]:>16s}: {100 * value.item():.2f}%")
# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)
# image_features = torch.cat(x, image_features)
# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
global_feat = torch.from_numpy(global_feat).to('cuda').to(torch.float16)
global_feat /= global_feat.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(30)
similarity = (100.0 * global_feat @ text_features.T).softmax(dim=-1)
# values, indices = similarity[0].topk(30)
# # Print the result
# # print("\nTop predictions:\n")
# # for value, index in zip(values, indices):
# #     print(f"{classes['1'][index]:>16s}: {100 * value.item():.2f}%")
#
print("\nTop predictions for global:\n")
for value, index in zip(values, indices):
    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")