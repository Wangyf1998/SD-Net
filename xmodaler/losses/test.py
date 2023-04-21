import torch
import torch.nn.functional as F


class NormalizedSoftmaxCELoss(torch.nn.Module):
    def __init__(self):
        super(NormalizedSoftmaxCELoss, self).__init__()

    def forward(self, input, target, label_embeddings):
        input_norm = F.normalize(input, p=2, dim=1)
        label_embeddings_norm = F.normalize(label_embeddings, p=2, dim=1)

        # Softmax Cross Entropy Loss
        ce_loss = F.cross_entropy(input, target)

        # Cosine similarity loss
        cosine_similarity = F.cosine_similarity(input_norm, label_embeddings_norm)
        cosine_loss = (1 - cosine_similarity).mean()

        # Combine losses
        loss = ce_loss + cosine_loss

        return loss
