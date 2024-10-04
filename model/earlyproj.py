import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional
from types import MethodType
from .base import DualEncoderModel
from .preprocess import targetpad_transform

class EarlyProjection(DualEncoderModel):
    def __init__(self, path="clip-vit-large-patch14"):
        # load CLIP model & CLIP tokenizer
        super().__init__()
        self.model = CLIPModel.from_pretrained(path, local_files_only=True)
        self.model.vision_model.encoder.forward = MethodType(encoder_forward_llf, self.model.vision_model.encoder)
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


def encoder_forward_llf(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    hidden_states = inputs_embeds

    for idx, encoder_layer in enumerate(self.layers):

        if idx == len(self.layers) - 1: #
            continue                    #

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                encoder_layer.__call__,
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions,
            )
        else:
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions=output_attentions,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)
    
    if output_hidden_states:
        encoder_states = encoder_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
    return BaseModelOutput(
        last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
    )

