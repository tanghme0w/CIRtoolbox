import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer


class CIRCombinerModel(nn.Module):
    def __init__(self, model: CLIPModel, tokenizer: CLIPTokenizer, combiner):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.combiner = combiner
    

    def get_query_feature(self, img, text):
        input_ids = self.tokenizer(text, padding='max_length', return_tensors='pt').input_ids
        text_features = self.model.get_text_features(input_ids)
        text_features = nn.functional.normalize(text_features)
        img_features = self.model.get_image_features(img)
        img_features = nn.functional.normalize(img_features)
        return self.combiner(text_features, img_features)
    
    
    def get_target_feature(self, img):
        return nn.functional.normalize(self.model.get_image_features(img))
