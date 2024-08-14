from .base import DualEncoderModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from  .MHTransformer import Transformer
from eva_clip import create_model_and_transforms, get_tokenizer
import os
from functools import partial
from torch.utils.checkpoint import checkpoint
from types import MethodType


def custom_visual_forward(self, x):

    x = self.patch_embed(x)
    batch_size, seq_len, _ = x.size()

    cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    x = torch.cat((cls_tokens, x), dim=1)
    if self.pos_embed is not None:
        x = x + self.pos_embed
    x = self.pos_drop(x)

    # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
    if os.getenv('RoPE') == '1':
        if self.training and not isinstance(self.patch_dropout, nn.Identity):
            x, patch_indices_keep = self.patch_dropout(x)
            self.rope.forward = partial(self.rope.forward, patch_indices_keep=patch_indices_keep)
        else:
            self.rope.forward = partial(self.rope.forward, patch_indices_keep=None)
            x = self.patch_dropout(x)
    else:
        x = self.patch_dropout(x)

    rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
    for blk in self.blocks:
        if self.grad_checkpointing:
            x = checkpoint(blk, x, (rel_pos_bias,))
        else:
            x = blk(x, rel_pos_bias=rel_pos_bias)

    global_feat = self.head(self.norm(x)[:, 0])

    return global_feat, x[:, 1:]

class FineGrainModel(DualEncoderModel):
    def __init__(
            self,
            topk=12, 
            epsilon=0.05, 
            img_patch_dim=1024, 
            token_feat=768, 
            tf_head=1, 
            tf_layer=3,
            num_k=24,
            model_name="EVA02-CLIP-L-14",
            pretrained="../eva-clip/EVA02_CLIP_L_psz14_s4B.pt"
        ):
        super().__init__()
        self.topk = topk
        self.num_k = num_k
        self.epsilon = epsilon
        self.local_atte_fc = nn.Sequential(nn.Linear(img_patch_dim, token_feat), nn.Sigmoid())
        self.transformer = Transformer(
            dim_self=img_patch_dim, 
            num_heads=tf_head, 
            dim_ref=img_patch_dim,
            num_layers=tf_layer
            )
        self.templates = nn.Parameter(torch.randn(1, num_k, img_patch_dim))
        self.model, _, self.preprocess = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
        self.tokenizer = get_tokenizer(model_name)
        self.model.visual.custom_visual_forward = MethodType(custom_visual_forward, self.model.visual)

    def query_forward(self, img, text):
        text = self.tokenizer(text).to(self.di.device)
        img = img.to(self.di.device)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        img_features = self.model.encode_image(img)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        return nn.functional.normalize(text_features + img_features)

    def target_forward(self, img):
        img = img.to(self.di.device)
        target_features = self.model.encode_image(img)
        return nn.functional.normalize(target_features)
    
    def fine_grained_query_forward(self, img, text):
        text = self.tokenizer(text).to(self.di.device)
        img = img.to(self.di.device)
        text_features = self.model.encode_text(text)
        text_global_feat = text_features / text_features.norm(dim=-1, keepdim=True)
        img_global_feat, img_local_tokens = self.model.visual.custom_visual_forward(img)
        img_global_feat = img_global_feat / img_global_feat.norm(dim=-1, keepdim=True)
        query_global_feat = nn.functional.normalize(text_global_feat + img_global_feat)
        image_fine_grained_feat = self.get_img_local_attr_feats(img_global_feat, img_local_tokens)[0]
        query_fine_grained_feat = self.get_img_local_attr_feats(query_global_feat, img_local_tokens)[0]
        return image_fine_grained_feat, query_fine_grained_feat

    def fine_grained_target_forward(self, img):
        img = img.to(self.di.device)
        target_global_feat, target_local_tokens = self.model.visual.custom_visual_forward(img)
        target_global_feat = target_global_feat / target_global_feat.norm(dim=-1, keepdim=True)
        target_fine_grained_feat = self.get_img_local_attr_feats(target_global_feat, target_local_tokens)[0]
        return target_fine_grained_feat

    def get_latent_local_attributes_feats(self, featuremap):
        """ 
        Borrowed from https://github.com/ZiChao111/FTI4CIR
        """
        batch_size = featuremap.shape[0]
        feature_dim = featuremap.shape[2]

        initial_templates = self.templates.expand(batch_size, self.num_k, feature_dim)
        cat_feature = torch.cat([initial_templates, featuremap], dim=1)
        latent_local_feats = self.transformer(cat_feature, mask=None)[:, :self.num_k, :]
        latent_local_feats = self.local_atte_fc(latent_local_feats)

        return latent_local_feats
    
    def get_img_local_attr_feats(self, global_feat, patch_token):
        """ 
        Borrowed from https://github.com/ZiChao111/FTI4CIR
        """
        bs = patch_token.shape[0]  # [bs, 257, 1024]
        latent_local_feats = self.get_latent_local_attributes_feats(patch_token)

        # Preliminary screening based on attention score
        attention_weights = torch.matmul(latent_local_feats, global_feat.unsqueeze(dim=2)).squeeze(dim=2)
        attention_weights = F.softmax(attention_weights, dim=1)

        local_attr_num = []
        sorted_indices = torch.argsort(attention_weights, dim=1, descending=True)
        sorted_indices = sorted_indices[:, :self.topk]
        selected_local_feats = []

        for i in range(bs):
            mask = attention_weights[i] > self.epsilon
            non_indices = torch.nonzero(mask).squeeze()
            num_r = non_indices.numel() if non_indices.numel() < self.topk else self.topk
            if num_r < 1:
                num_r = 1
            # Ensure the order of attribute features
            select_indices = sorted_indices[i][:num_r]
            select_indices = torch.sort(select_indices, dim=0).values
            select_id = torch.cat((select_indices, sorted_indices[i][num_r:]), dim=0)
            local_attr_num.append(num_r)
            selected_local_feats.append(latent_local_feats[i, select_id, :])

        selected_local_feats = torch.stack(selected_local_feats, dim=0)

        return F.normalize(selected_local_feats, dim=-1), local_attr_num