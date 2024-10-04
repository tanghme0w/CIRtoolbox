import torch.nn as nn
from .base import DualEncoderModel
from transformers import CLIPModel, CLIPTokenizer
from .preprocess import targetpad_transform
from eva_clip import create_model_and_transforms, get_tokenizer


class CLIPSumModel(DualEncoderModel):
    def __init__(self, path="clip-vit-large-patch14"):
        # load CLIP model & CLIP tokenizer
        super().__init__()
        self.model = CLIPModel.from_pretrained(path, local_files_only=True)
        self.tokenizer = CLIPTokenizer.from_pretrained(path, local_files_only=True)
        self.preprocess = targetpad_transform()

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