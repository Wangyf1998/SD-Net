####################################
# 这个文件作为所有filter的基类使用，直接把global feature作为clip的视觉输入， text prompt作为文本输入，然后训练最后一个全连接层
import torch
import clip
from torch import nn
import torch.nn.functional as F
import json
import pickle
from itertools import chain

from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import CLIP_REGISTRY

@CLIP_REGISTRY.register()
class PromptClipFilter(nn.Module):
    @configurable
    def __init__(
            self,



    ):
        super(PromptClipFilter, self).__init__(
        )
        # self.fusion_layer = fusion_layer
        self.v_linear = nn.Linear(512, 512, bias=True)
        self.t_linear = nn.Linear(512, 512)

    @classmethod
    def from_config(cls, cfg):

        return {

        }
    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        # 读取数据
        ret = {}
        model, _ = clip.load('ViT-B/32', 'cuda')
        vfeats = batched_inputs[kfg.GLOBAL_FEATS]
        emo_label = batched_inputs[kfg.G_EMO_LABEL]
        attr = batched_inputs[kfg.G_ATTR_IDS]
        learned_token = batched_inputs.get(kfg.ENCODER_TOKEN, None)
        batch_size = vfeats.size(0)
        emo_list = list(chain(*emo_label))
        emo_table = ["amusing", "awesome", "content", "excited", "angry", "disgusting", "fearful", "sad", ""]

        # with open("/home/wyf/open_source_dataset/for_debug/4.5/concept_pool.json", 'r') as f:
        #     concept_pool = json.loads(f)
        concept_pool = json.load(open("/home/wyf/open_source_dataset/for_debug/4.5/concept_pool.json", 'r'))
        concept_pool_ids = pickle.load(open("/home/wyf/open_source_dataset/for_debug/4.5/concept_pool_ids.pkl", 'rb'), encoding='bytes')
        prompt_list = []     # 创建一个不同emotion构成的prompt
        for i in range(batch_size):
            label = emo_list[i]
            sent = "It's a {} painting of ".format(emo_table[label])
            if label in {0, 1, 2, 3}:
                prompt = torch.cat([clip.tokenize(f"{sent}{c}") for c in concept_pool['0']]).to('cuda')
            elif label in {4, 5, 6, 7}:
                prompt = torch.cat([clip.tokenize(f"{sent}{c}") for c in concept_pool['1']]).to('cuda')
            else:
                prompt = torch.cat([clip.tokenize(f"{sent}{c}") for c in concept_pool['2']]).to('cuda')
            prompt_list.append(prompt)
        text_inputs = torch.cat([])
        text_feature_list = []  # 创建一个由text feature构成的list
        attr_pred = []          # 将CLIP从concept pool中筛选的concept存入attr_pred中
        for text in prompt_list:
            with torch.no_grad():
                text_feature = model.encode_text(text)
                text_feature /= text_feature.norm(dim=-1, keepdim=True)
                text_feature_list.append(text_feature)

        # image_features = self.fusion_layer(vfeats)
        image_features = self.v_linear(vfeats)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.to(torch.float16)
        sim_list = []
        attr_pred = []
        for i in range(batch_size):
            attr = []
            attr_ids = []
            similarity = (100.0 * image_features[i, :] @ text_feature_list[i].T).softmax(dim=-1)
            sim_list.append(similarity)
            values, indices = similarity.topk(10)
            if emo_list[i] in {0,1,2,3}:
                for index in indices:
                    attr.append(concept_pool['0'][index])
                    attr_ids.append(concept_pool_ids[0][0][index])
            elif emo_list[i] in {4, 5, 6, 7}:
                for index in indices:
                    attr.append(concept_pool['1'][index])
                    attr_ids.append(concept_pool_ids[1][0][index])
            else:
                for index in indices:
                    attr.append(concept_pool['2'][index])
                    attr_ids.append(concept_pool_ids[2][0][index])
            attr_pred.append(attr)


        # text_features = torch.cat(text_feature_list, dim=0)
        # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # _, indices = similarity[0].topk(5)
        ret.update({ kfg.PRED_ATTR: attr_pred

        })
        return ret



# Download the dataset
# cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs
# image, class_id = cifar100[3637]
# #TODO：我们需要提取数据集中的concepts作为label
# image_input = preprocess(image).unsqueeze(0).to(device)
# sentiment = []
# text_inputs = torch.cat([clip.tokenize(f"It's a sentiment painting of {c}") for c in cifar100.classes]).to(device)
#
# # Calculate features
# with torch.no_grad():
#     image_features = model.encode_image(image_input)
#     text_features = model.encode_text(text_inputs)
#
# # Pick the top 5 most similar labels for the image
# image_features /= image_features.norm(dim=-1, keepdim=True)
# text_features /= text_features.norm(dim=-1, keepdim=True)
# # image feature前是否可以加一个可训练的MLP
# similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
# values, indices = similarity[0].topk(5)
#
# # Print the result
# print("\nTop predictions:\n")
# for value, index in zip(values, indices):
#     print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")