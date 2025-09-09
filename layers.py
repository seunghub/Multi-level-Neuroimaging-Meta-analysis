import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import mp_utils
import lorentz as L


class MLP(nn.Module):
    def __init__(self, input_size, output_size, inner_size, num_layers,
                 activation_func=nn.GELU(), dropout=0.1):
        super(MLP, self).__init__()

        # Create a list to hold the layers
        layers = [
            nn.Linear(input_size, inner_size),
            nn.LayerNorm(inner_size),
            activation_func,
            nn.Dropout(dropout),

        ]
        # Hidden layers
        for _ in range(num_layers-1):
            layers.append(nn.Linear(inner_size, inner_size))
            nn.LayerNorm(inner_size),
            layers.append(activation_func)
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(inner_size, output_size))

        # Combine all layers into a Sequential module
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class TmpHead(nn.Module):
    def __init__(
        self,
        dim,
        dropout,
    ):
        super().__init__()
        self.gelu = nn.GELU()
        self.fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        out = self.fc(x)
        # out = self.gelu(out)
        # out = self.dropout(out)
        # out = x + out
        out = self.layer_norm(out)
        return out
    
class ResidualHead(nn.Module):
    def __init__(
        self,
        dim,
        dropout,
    ):
        super().__init__()
        self.gelu = nn.GELU()
        self.fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        out = self.fc(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = x + out
        out = self.layer_norm(out)
        return out


class ProjectionHead(nn.Module):
    """Taken from https://www.kaggle.com/code/moeinshariatnia/openai-clip-simple-implementation"""
    def __init__(
        self,
        embedding_dim,
        output_dim,
        dropout,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, output_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class ClipModel(nn.Module):
    def __init__(self, image_model, text_model, logit_scale=np.log(1/0.07), logit_bias=None):
        super().__init__()

        self.image_model = image_model
        self.text_model = text_model
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale)
        self.logit_bias = (
            nn.Parameter(torch.ones([]) * logit_bias) if logit_bias else None
        )

    def encode_image(self, image):  # DiFuMo
        return self.image_model(image)

    def encode_text(self, text):  # Embeddings
        return self.text_model(text)

    def forward(self, image, text):
        image_embeddings = self.encode_image(image)
        # print(f"image_embeddings shape: {image_embeddings.shape}")
        
        text_embeddings = self.encode_text(text)
        # print(f"text_embeddings shape: {text_embeddings.shape}")

        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        return image_embeddings, text_embeddings

class MeruModel(ClipModel):
    def __init__(
            self, 
            image_model, 
            text_model, 
            embed_dim: int,
            curv_init: float = 1.0,
            learn_curv: bool = True,
            entail_weight: float = 0.0,
            pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
            pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
            device: str = 'cuda',
            logit_scale=np.log(1/0.07), 
            logit_bias=None
    ):
        """
        Args:
            curv_init: Positive scalar that denotes negative Hyperboloid curvature.
            learn_curv: Whether to learn the curvature parameter during training.
            entail_weight: Weight for the entailment loss component.
        """

        super().__init__(image_model, text_model, logit_scale, logit_bias)

        self.device = device

        self.image_model = image_model
        self.text_model = text_model
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale)
        self.logit_bias = (
            nn.Parameter(torch.ones([]) * logit_bias) if logit_bias else None
        )

        # Initialize curvature parameter. Hyperboloid curvature will be `-curv`.
        self.curv = nn.Parameter(
            torch.tensor(curv_init).log(), requires_grad=learn_curv
        )

        # When learning the curvature parameter, restrict it in this interval to
        # prevent training instability.
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }
        self.entail_weight = entail_weight

        # Learnable scalars to ensure that image/text features have an expected
        # unit norm before exponential map (at initialization).
        self.visual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.textual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
    
    def encode_image(self, image: torch.Tensor, project: bool):
        """
        Args:
            image: Image batch in BCHW format, with pixel values in `[0, 1]`.
            project: Lift features from the encoder onto the Hyperboloid.

        Returns:
            Batch of image features of shape `(B, visual.width)`.
        """

        # Get Euclidean features from the encoder (without L2 normalization).
        image_feats = super().encode_image(image)

        # These features are space components of embeddings in the tangent
        # space of the Hyperboloid origin (which is Euclidean). Apply projection.
        if project:
            image_feats = image_feats * self.visual_alpha.exp()
            image_feats = L.exp_map0(image_feats, self.curv.exp())
        return image_feats
    
    def encode_text(self, text: list[torch.Tensor], project: bool):
        """
        Args:
            text: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
            project: Lift features from the encoder onto the Hyperboloid.
        """

        # Get Euclidean features from the encoder (without L2 normalization).
        text_feats = super().encode_text(text)

        if project:
            text_feats = text_feats * self.textual_alpha.exp()
            text_feats = L.exp_map0(text_feats, self.curv.exp())

        return text_feats

    def forward(
        self, images: torch.Tensor, tokens: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """

        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()

        # Clamp scaling factors such that they do not up-scale the feature norms.
        # Once `exp(scale) = 1`, they can simply be removed during inference.
        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)

        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        # all_image_feats = mp_utils.gather_across_processes(image_feats)
        # all_text_feats = mp_utils.gather_across_processes(text_feats)

        # # shape: (batch_size * world_size, embed_dim)
        # all_image_feats = torch.cat(all_image_feats, dim=0)
        # all_text_feats = torch.cat(all_text_feats, dim=0)

        # # Compute all necessary loss components. We enclose the entire block with
        # # autocast to force a higher floating point precision.
        # with torch.autocast(self.device.type, dtype=torch.float32):
        #     # Compute logits for contrastive loss.
        image_logits = -L.pairwise_dist(image_feats, text_feats, _curv)
        text_logits = -L.pairwise_dist(text_feats, image_feats, _curv)

        #     # Compute cross entropy loss: we compute log probabilities and take the
        #     # diagonal elements as targets: image[i] should match text[i] in batch.
        #     # Shift the targets according to rank of GPU process (we assume that all
        #     # GPU processes have the same local batch size).
        #     batch_size = image_feats.shape[0]
        targets = torch.arange(image_logits.shape[0], device=image_logits.device)
        #     targets = targets + batch_size * self._rank

        #     # Clamp temperature such that logits are not scaled more than 100x.
        #      # ln(100) = ~4.6052
        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
        _scale = self.logit_scale.exp()

        # Hyperbolic entailment loss: text should entail matching image.
        _angle = L.oxy_angle(text_feats, image_feats, _curv)
        _aperture = L.half_aperture(text_feats, _curv)
        entailment_loss = torch.clamp(_angle - _aperture, min=0).mean()

        if self.entail_weight > 0:
            entailment_loss = self.entail_weight * entailment_loss

        entailment_loss = entailment_loss
        contrastive_loss = 0.5 * (
                nn.functional.cross_entropy(_scale * image_logits, targets)
                + nn.functional.cross_entropy(_scale * text_logits, targets)
            )

        # print(entailment_loss.item(), contrastive_loss.item())
        return {'image_embedding': image_feats,
                'text_embedding': text_feats,
                'entailment_loss': entailment_loss,
                'contrastive_loss': contrastive_loss
                }



class OurModel(MeruModel):
    def forward(
        self, images: torch.Tensor, tokens: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """

        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()

        # Clamp scaling factors such that they do not up-scale the feature norms.
        # Once `exp(scale) = 1`, they can simply be removed during inference.
        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)

        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        #     # Clamp temperature such that logits are not scaled more than 100x.
        #      # ln(100) = ~4.6052
        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
        _scale = self.logit_scale.exp()

        # # Hyperbolic entailment loss: text should entail matching image.
        _alpha = L.pairwise_oxy_angle(text_feats, image_feats, _curv)
        _alpha = -torch.abs(_alpha)
        targets = torch.arange(_alpha.shape[0], device=_alpha.device)
        contrastive_loss = nn.functional.cross_entropy(_scale * _alpha, targets)

        entailment_loss = L.centroid_loss(image_feats, 2.0, _curv) + L.centroid_loss(text_feats, 0.5, _curv)
        activated_ROIs = (images>0.005).sum(-1)
        image_dists = torch.sqrt(1/_curv+torch.sum(image_feats**2,dim=-1).unsqueeze(-1))
        ROIs_gap = (activated_ROIs[:,None]-activated_ROIs[None,:])>10
        dists_gap = torch.log(image_dists/image_dists.T); dists_gap[dists_gap<=0]=0
        hierachical_loss = 30 * (ROIs_gap*dists_gap).sum(-1).mean()

        contrastive_loss = contrastive_loss + hierachical_loss

        if self.entail_weight > 0: # centroid loss 0.5 p and q 2 0.5
            entailment_loss = self.entail_weight * entailment_loss

        return {'image_embedding': image_feats,
                'text_embedding': text_feats,
                'entailment_loss': entailment_loss,
                'contrastive_loss': contrastive_loss
                }