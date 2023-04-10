# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import math
import torch
from torch import nn
from itertools import  chain

from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from ..layers.bert import BertLayer, BertGenerationLayer
from .build import ENCODER_REGISTRY

__all__ = ["COSNetEncoder"]

@ENCODER_REGISTRY.register()
class COSNetEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        bert_layers,
        semcomphder_layers,
        hidden_size: int,
        num_hidden_layers: int,
        num_semcomphder_layers: int,
        slot_size: int,
        num_classes: int,
        max_pos: int
    ):
        super(COSNetEncoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.num_semcomphder_layers = num_semcomphder_layers
        
        self.layers = bert_layers
        self.decoder_enc_layers = semcomphder_layers
        self.num_classes = num_classes
        self.slot_size = slot_size
        self.max_pos_len = max_pos

        self.query_pred = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 3)
        )
        self.semantics_pred = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes+1)   
        )

        self.gvfeat_embed = nn.Sequential(
            nn.Linear(hidden_size * (num_hidden_layers + 1), hidden_size),
            torch.nn.LayerNorm(hidden_size)
        )
        self.embeddings = nn.Sequential(
            nn.Embedding(num_classes, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
        )

        self.slot_embeddings = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
        )

        self.embeddings_emo = nn.Embedding(20985, 512)

        self.slot = nn.Parameter(torch.FloatTensor(1, slot_size, hidden_size))
        nn.init.xavier_uniform_(self.slot)

        self.softmax = nn.Softmax(dim=-1)
        self.position = nn.Parameter(torch.FloatTensor(self.max_pos_len, hidden_size))
        nn.init.xavier_uniform_(self.position)

    @classmethod
    def from_config(cls, cfg):
        bert_layers = nn.ModuleList(
            [BertLayer(cfg) for _ in range(cfg.MODEL.BERT.NUM_HIDDEN_LAYERS)]
        )
        semcomphder_layers = nn.ModuleList(
            [BertGenerationLayer(cfg) for _ in range(cfg.MODEL.COSNET.NUM_SEMCOMPHDER_LAYERS)]
        )
        return {
            "bert_layers": bert_layers,
            "semcomphder_layers": semcomphder_layers,
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "num_hidden_layers": cfg.MODEL.BERT.NUM_HIDDEN_LAYERS,
            "num_semcomphder_layers": cfg.MODEL.COSNET.NUM_SEMCOMPHDER_LAYERS,
            "slot_size": cfg.MODEL.COSNET.SLOT_SIZE,
            "num_classes": cfg.MODEL.COSNET.NUM_CLASSES,
            "max_pos": cfg.MODEL.COSNET.MAX_POS
        }

    @classmethod
    def add_config(cls, cfg):
        cfg.MODEL.COSNET = CN()
        cfg.MODEL.COSNET.NUM_SEMCOMPHDER_LAYERS = 0
        cfg.MODEL.COSNET.SLOT_SIZE = 0
        cfg.MODEL.COSNET.NUM_CLASSES = 906
        cfg.MODEL.COSNET.MAX_POS = 26
        cfg.MODEL.COSNET.FILTER_WEIGHT = 1.0
        cfg.MODEL.COSNET.RECONSTRUCT_WEIGHT = 0.1

    def forward(self, batched_inputs, mode=None):
        ret = {}

        if mode == None or mode == 'v':
            vfeats = batched_inputs[kfg.ATT_FEATS]
            batch_size = vfeats.shape(0)
            ext_vmasks = batched_inputs[kfg.EXT_ATT_MASKS]
            ext_vmasks = torch.cat([ext_vmasks[:, :, :, 0:1], ext_vmasks], dim=-1)
            ret.update({kfg.EXT_ATT_MASKS: ext_vmasks})
            # memory_dict = {}
            # if batched_inputs['ENCODER_TOKEN'] is None:
            #     emotion_token = torch.zeros(9, 512)
                #0-3:positive，4-7：negative
            # emo_label = batched_inputs['G_EMO_LABEL']
            # emo_list = list(chain(*emo_label))
            gfeats = []
            gfeats.append(vfeats[:, 0])
            encoder_vfeats = vfeats

            for layer_module in self.layers:
                encoder_vfeats, _ = layer_module(encoder_vfeats, ext_vmasks)
                gfeats.append(encoder_vfeats[:, 0])
            gfeats = torch.cat(gfeats, dim=-1)
            gfeats = self.gvfeat_embed(gfeats)
            encoder_vfeats = torch.cat([gfeats.unsqueeze(1), encoder_vfeats[:, 1:]], dim=1)
            ret.update({kfg.ATT_FEATS: encoder_vfeats })


            # semantics_ids = batched_inputs[kfg.SEMANTICS_IDS]
            attr_mask = batched_inputs[kfg.ATTR_MASK]
            attr_mask = (1.0 - attr_mask) * - 10000.0
            attr_mask = attr_mask.unsqueeze(1).unsqueeze(2)
            ret.update({kfg.EXT_ATTR_MASK: attr_mask})

            query = batched_inputs.get(kfg.QUERY_EMBED, None)
            if query is None:
                query = nn.Parameter(torch.FloatTensor(1, query_size, hidden_size))
                nn.init.xavier_uniform_(query)
                query_embed = self.slot_embeddings(query)
                query_embed = query_embed.expand(batch_size, query_embed.shape[1], query_embed.shape[2])
            else:
                query_embed = query
            # query_mask = torch.ones((query_embed.shape[0], query_embed.shape[1]), device=query_embed.device).to(dtype=next(self.parameters()).dtype)
            query_mask = None
            for layer_module in self.decoder_enc_layers:
                query_embed = layer_module(semantics_embed, encoder_vfeats, query_mask, ext_vmasks)
            query_output = self.query_pred(query_embed)

            ret.update({
                kfg.QUERY_EMBED:query_embed,
                kfg.QUERY_OUTPUT: query_output
            })




            # semantics_embed = self.embeddings(semantics_ids)
            # slot_embed = self.slot_embeddings(self.slot)
            # slot_embed = slot_embed.expand(semantics_embed.shape[0], slot_embed.shape[1], slot_embed.shape[2])
            # 将slot_embed扩展至于semantic_embed相同的维度
            # semantics_embed = torch.cat([slot_embed, semantics_embed], dim=1)
            #
            # slot_mask = torch.ones((semantics_embed.shape[0], slot_embed.shape[1]), device=slot_embed.device).to(dtype=next(self.parameters()).dtype)
            # semantics_mask = torch.cat([slot_mask, semantics_mask], dim=1)
            # #
            # semantics_mask = (1.0 - semantics_mask) * -10000.0
            # semantics_mask = semantics_mask.unsqueeze(1).unsqueeze(2)
            #
            # for layer_module in self.decoder_enc_layers:
            #     semantics_embed = layer_module(semantics_embed, encoder_vfeats, semantics_mask, ext_vmasks)
            #
            # semantics_pred = self.semantics_pred(semantics_embed)
            #
            # ret.update({
            #     kfg.SEMANTICS_PRED: semantics_pred,
            #     kfg.SEMANTICS_FEATS: semantics_embed,
            #     kfg.EXT_SEMANTICS_MASKS: semantics_mask,
            # })
            #
            # semantics_pos_pred = semantics_embed @ self.position.t()
            # semantics_pos_prob = self.softmax(semantics_pos_pred)
            # position = semantics_pos_prob @ self.position
            # semantics_embed = semantics_embed + position
            # ret.update({ kfg.SEMANTICS_FEATS: semantics_embed, kfg.SEMANTICS_POS_PRED: semantics_pos_pred })
        return ret
