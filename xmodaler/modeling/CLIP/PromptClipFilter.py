####################################
# 这个文件作为所有filter的基类使用，直接把global feature作为clip的视觉输入，text prompt作为文本输入
import torch
import clip
from torch import nn
import torch.nn.functional as F
import json
import pickle
from itertools import chain
import numpy as np

from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import CLIP_REGISTRY
from torch.nn.utils.rnn import pad_sequence

@CLIP_REGISTRY.register()
class PromptClipFilter(nn.Module):
    @configurable
    def __init__(
            self,
):
        super(PromptClipFilter, self).__init__(
        )
        # self.fusion_layer = fusion_layer
        self.before_linear = torch.nn.Sequential(
            nn.Linear(3 * 224 * 224, 1024),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.after_linear = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.embeddings = nn.Sequential(
            nn.Embedding(1000, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.1))
        # self.mask_prompt = nn.parameter()

        self.t_linear = nn.Linear(512, 512)
        self.use_image_mask = True                        # 一组或多组参数直接加在image feature上，类似于mask的prompt
        self.use_random_prompt = True                     # 提供一组随机初始化的query，经由一个embedding后与image feature混合
        self.use_word_prompt = True                       # 提供一组以情感倾向的编码作为query的初始值的prompt
        self.use_id_prompt = True                         # 提供一组以id中的单词作为query初始值的prompt

        self.use_manual_prompt = True                     # 提供一组人工生成的prompt作为text prompt
        self.use_auto_prompt = True                       # 提供一组经由embedding后的prompt作为text prompt

        self.use_MLP = True                               # 允许image feature先经过一层MLP以缩小gap
        self.use_after_MLP = True                         # 允许编码后的image feature再经过一层MLP以提升性能




    @classmethod
    def from_config(cls, cfg):

        return {

        }
    @classmethod
    def add_config(cls, cfg):
        pass

    def get_label(self, pool_ids, g_concepts, emo_list, batch_size, art_style):
        final_label = []
        for i in range(batch_size):
            emo = emo_list[i]
            style = int(art_style[i])
            use_pool = pool_ids[style]
            if emo in {0,1,2,3}:
                concept_label = torch.zeros(use_pool['0'].size(0))
                concept_label[torch.isin(use_pool['0'], g_concepts[i, :])] = 1
                final_label.append(concept_label)
            elif emo in {4,5,6,7}:
                concept_label = torch.zeros(use_pool['1'].size(0))
                concept_label[torch.isin(use_pool['1'], g_concepts[i, :])] = 1
                final_label.append(concept_label)
        torch.cat(final_label, dim=0)
        return final_label

    def load_data(self):
        pool_0 = json.load(open("/home/wyf/open_source_dataset/artemis_dataset/4.19/pool_1.json", 'r'))
        pool_1 = json.load(open("/home/wyf/open_source_dataset/artemis_dataset/4.19/pool_2.json", 'r'))
        pool_2 = json.load(open("/home/wyf/open_source_dataset/artemis_dataset/4.19/pool_3.json", 'r'))
        pool_3 = json.load(open("/home/wyf/open_source_dataset/artemis_dataset/4.19/pool_4.json", 'r'))
        pool_ids_0 = pickle.load(
            open("/home/wyf/open_source_dataset/artemis_dataset/4.19/dict_1.pkl", 'rb'),
            encoding='bytes')
        pool_ids_1 = pickle.load(
            open("/home/wyf/open_source_dataset/artemis_dataset/4.19/dict_2.pkl", 'rb'),
            encoding='bytes')
        pool_ids_2 = pickle.load(
            open("/home/wyf/open_source_dataset/artemis_dataset/4.19/dict_3.pkl", 'rb'),
            encoding='bytes')
        pool_ids_3 = pickle.load(
            open("/home/wyf/open_source_dataset/artemis_dataset/4.19/dict_4.pkl", 'rb'),
            encoding='bytes')
        for k in pool_ids_0:
            pool_ids_0[k] = torch.tensor(pool_ids_0[k]).to('cuda')
        for k in pool_ids_1:
            pool_ids_1[k] = torch.tensor(pool_ids_1[k]).to('cuda')
        for k in pool_ids_2:
            pool_ids_2[k] = torch.tensor(pool_ids_2[k]).to('cuda')
        for k in pool_ids_3:
            pool_ids_3[k] = torch.tensor(pool_ids_3[k]).to('cuda')
        return pool_0, pool_1, pool_2, pool_3, pool_ids_0, pool_ids_1, pool_ids_2, pool_ids_3

    def pad(self, concepts_label, clip_pred):
        label_output = pad_sequence(concepts_label, batch_first=True, padding_value=0)
        clip_output = pad_sequence(clip_pred, batch_first=True, padding_value=0)
        return label_output, clip_output

    def forward(self, batched_inputs):
        # 读取数据
        ret = {}
        model, _ = clip.load('RN101', 'cuda')
        vfeats = batched_inputs[kfg.GLOBAL_FEATS]
        emo_label = batched_inputs[kfg.G_EMO_LABEL]
        art_style = batched_inputs[kfg.ART_STYLE]
        g_concepts = batched_inputs[kfg.G_ATTR_IDS]
        batch_size = vfeats.size(0)
        emo_list = list(chain(*emo_label))
        pool_0, pool_1, pool_2, pool_3, pool_ids_0, pool_ids_1, pool_ids_2, pool_ids_3 = self.load_data()
        pool_ids = [pool_ids_0, pool_ids_1, pool_ids_2, pool_ids_3]
        pool = [pool_0, pool_1, pool_2, pool_3]
        # emo_table = ["amusing", "awesome", "content", "excited", "angry", "disgusting", "fearful", "sad", ""]
        prompt_list = []     # 创建一个不同emotion构成的prompt
        concepts_label = self.get_label(pool_ids, g_concepts, emo_list, batch_size, art_style)
        for i in range(batch_size):
            label = emo_list[i]
            style = int(art_style[i])
            sent = "It's a painting of "
            if label in {0,1,2,3}:
                prompt = torch.cat([clip.tokenize(f"{sent}{c}") for c in pool[style]['0']]).to('cuda')
                prompt_list.append(prompt)
            elif label in {4,5,6,7}:
                prompt = torch.cat([clip.tokenize(f"{sent}{c}") for c in pool[style]['1']]).to('cuda')
                prompt_list.append(prompt)
        # prompt_list = torch.cat(prompt_list, dim=0)
        text_feature_list = []
        with torch.no_grad():
            for i in range(batch_size):
                text_feature = model.encode_text(prompt_list[i])
                text_feature /= text_feature.norm(dim=-1, keepdim=True)
                text_feature_list.append(text_feature)
        text_feature = pad_sequence(text_feature_list, batch_first=True, padding_value=0)
        if self.after_linear:
            vfeats = self.after_linear(vfeats)
        vfeats /= vfeats.norm(dim=-1, keepdim=True)
        # vfeats = self.v_linear(vfeats)
        vfeats = vfeats.to(torch.float16)
        similarity = []
        #TODO:similarity是对这个batch内的所有样本在对应concept_pool上的表格，因此，要生成对应样本的label，即一个独热向量，然后计算loss
        for i in range(batch_size):
            simi = (100.0 * vfeats[i, :] @ text_feature_list[i].T).softmax(dim=-1)
            similarity.append(simi)
        clip_pred = similarity
        # similarity.append(sim)
        # similarity = torch.cat(similarity, dim=0)
        concept_pred = []
        concept_pred_ids = []

        for i in range(batch_size):
            values, indices = similarity[i].topk(5)
            style = int(art_style[i])
            if emo_list[i] in {0, 1, 2, 3}:
                pred = []
                pred_ids = []
                for index in indices:
                    pred_ids.append(pool_ids[style]['0'][index])
                    pred.append(pool[style]['0'][index])
                concept_pred.append(pred)
                concept_pred_ids.append(pred_ids)
            elif emo_list[i] in {4, 5, 6, 7}:
                pred = []
                pred_ids = []
                for index in indices:
                    pred_ids.append(pool_ids[style]['1'][index])
                    pred.append(pool[style]['1'][index])
                concept_pred.append(pred)
                concept_pred_ids.append(pred_ids)
        label_output, clip_output = self.pad(concepts_label, clip_pred)
        ret.update({kfg.CONCEPTS_PRED: concept_pred,
                    kfg.CONCEPTS_PRED_IDS: concept_pred_ids,
                    kfg.CONCEPTS_LABEL: label_output,
                    kfg.CLIP_PRED: clip_output
                    })
        return ret

