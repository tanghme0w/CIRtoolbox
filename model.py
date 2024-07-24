import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer
from typing import Optional, List
from PIL.Image import Image
import numpy as np


class DualEncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.di = nn.Parameter(torch.empty(0))  # device indicator

    def query_forward(self, img, text):
        raise NotImplementedError("query forward not implemented")

    def target_forward(self, img):
        raise NotImplementedError("target forward not implemented")


class CLIPSumModel(DualEncoderModel):
    def __init__(self, path="clip-vit-large-patch14"):
        # load CLIP model & CLIP tokenizer
        super().__init__()
        self.model = CLIPModel.from_pretrained(path, local_files_only=True)
        self.tokenizer = CLIPTokenizer.from_pretrained(path, local_files_only=True)
        
    def query_forward(self, img, text):
        input_ids = self.tokenizer(text, padding='max_length', return_tensors='pt').input_ids.to(self.di.device)
        img = img.to(self.di.device)
        text_features = self.model.get_text_features(input_ids)
        img_features = self.model.get_image_features(img)
        return nn.functional.normalize(text_features + img_features)

    def target_forward(self, img):
        img = img.to(self.di.device)
        target_features = self.model.get_image_features(img)
        return nn.functional.normalize(target_features)

