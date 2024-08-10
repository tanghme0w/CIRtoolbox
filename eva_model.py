from model import DualEncoderModel
from eva_clip import create_model_and_transforms, get_tokenizer
import torch.nn as nn


class CLIPSumModelEVA(DualEncoderModel):
    def __init__(self, path="clip-vit-large-patch14"):
        # load CLIP model & CLIP tokenizer
        super().__init__()
        model_name = "EVA02-CLIP-L-14" 
        pretrained = "../eva-clip/EVA02_CLIP_L_psz14_s4B.pt"
        self.model, _, self.preprocess = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
        self.tokenizer = get_tokenizer(model_name)
        
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
