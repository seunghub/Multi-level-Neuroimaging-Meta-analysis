import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import lorentz as L

class ClipLoss(nn.Module):
    def forward(self, image_embeddings, text_embeddings, logit_scale, *_):
        logits_per_image = logit_scale * image_embeddings @ text_embeddings.T
        logits_per_text = logit_scale * text_embeddings @ image_embeddings.T
        labels = torch.arange(len(logits_per_image), device=image_embeddings.device)

        return (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2
    

# class MeruLoss(nn.Module):
#     def forward(self, image_embeddings, text_embeddings, logit_scale, logit_bias, entailment_loss):
#         # print(logit_scale, image_embeddings, text_embeddings)
        
#         logits_per_image = logit_scale * image_embeddings @ text_embeddings.T
#         logits_per_text = logit_scale * text_embeddings @ image_embeddings.T
#         labels = torch.arange(len(logits_per_image), device=image_embeddings.device)

#         return (
#             F.cross_entropy(logits_per_image, labels)
#             + F.cross_entropy(logits_per_text, labels)
#         ) / 2 + entailment_loss

class MeruLoss(nn.Module):
    def forward(self, image_embeddings, text_embeddings, logit_scale, curvature, entailment_loss):
        # print(logit_scale, image_embeddings, text_embeddings)
        image_logits = -L.pairwise_dist(image_embeddings, text_embeddings, curvature)
        text_logits = -L.pairwise_dist(text_embeddings, image_embeddings, curvature)

        labels = torch.arange(len(image_logits), device=image_embeddings.device)

        return (
            F.cross_entropy(logit_scale*image_logits, labels)
            + F.cross_entropy(logit_scale*text_logits, labels)
        ) / 2 + entailment_loss