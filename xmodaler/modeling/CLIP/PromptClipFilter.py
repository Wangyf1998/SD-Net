import clip
from torch import nn
import torch.nn.functional as F

from xmodaler.config import configurable
from xmodaler.config import kfg
from .clipfilter import  ClipFilter
from .build import CLIP_REGISTRY

@CLIP_REGISTRY.register()
class PromptClipFilter(ClipFilter):
    @configurable
    def __init__(
            self,
            *,
            fusion_layer


    ):
        super(PromptClipFilter, self).__init__(
        )
        self.fusion_layer = fusion_layer
        self.model = clip.load('ViT-B/32', 'cuda')

    @classmethod
    def from_config(cls, cfg):
        return {
            "fusion_layer": cfg.fusion_layer
        }
    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        ret = {}
        vfeats = batched_inputs[kfg.GLOBAL_FEATS]
        emo_label = batched_inputs[kfg.EMO_LABEL]
        concept_pool = concept_pool
        # 后期填上concept pool。concept pool应该是一个dict，每一个情感键下有对应的编码过后的concept，作为label
        attr = batched_inputs[kfg.G_ATTR_IDS]
        emo_list = {"amusing", "awesome", "content", "excited", "angry", "disgusting", "fearful", "sad", ""}
        prompt = "It's a {} painting of ".format(emo_list[emo_label])
        text_inputs = torch.cat([clip.tokenize(f"{prompt}{c}") for c in concept_pool]).to(cuda)
        image_features = self.fusion_layer(vfeats)
        with torch.no_grad:
            text_features = self.model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # 把以上几行写进fusion layer里，要能选择多种soft prompt的拼接方式
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        _, indices = similarity[0].topk(5)
        ret.update({ kfg.PRED_ATTR: indices

        })
        return ret



# Download the dataset
# cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs
image, class_id = cifar100[3637]
#TODO：我们需要提取数据集中的concepts作为label
image_input = preprocess(image).unsqueeze(0).to(device)
sentiment = []
text_inputs = torch.cat([clip.tokenize(f"It's a sentiment painting of {c}") for c in cifar100.classes]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
# image feature前是否可以加一个可训练的MLP
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")