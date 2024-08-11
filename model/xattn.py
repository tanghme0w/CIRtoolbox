import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from typing import Optional
from types import MethodType
from data.preprocess import targetpad_transform
from .base import DualEncoderModel


class LastHiddenXAttnT2I(DualEncoderModel):
    def __init__(self, path="clip-vit-large-patch14"):
        # load CLIP model & CLIP tokenizer
        super().__init__()
        self.model = CLIPModel.from_pretrained(path, local_files_only=True)
        self.model.get_text_outputs = MethodType(get_text_outputs, self.model)
        self.model.get_image_features_XAttnT2I = MethodType(get_image_features_XAttnT2I, self.model)
        self.model.vision_model.CLIP_vit_forward_XAttnT2I = MethodType(CLIP_vit_forward_XAttnT2I, self.model.vision_model)
        self.model.vision_model.encoder.CLIP_encoder_forward_XAttnT2I = MethodType(CLIP_encoder_forward_XAttnT2I_LastHidden, self.model.vision_model.encoder)
        self.model.vision_model.encoder.xattn_layer = nn.MultiheadAttention(embed_dim=1024, num_heads=1, kdim=768, vdim=768, batch_first=True)
        self.model.vision_model.encoder.layer_norm = nn.LayerNorm(normalized_shape=(257, 1024))
        self.tokenizer = CLIPTokenizer.from_pretrained(path, local_files_only=True)

    def query_forward(self, img, text):
        input_ids = self.tokenizer(text, padding='max_length', return_tensors='pt').input_ids.to(self.di.device)
        img = img.to(self.di.device)
        text_features, text_last_hidden = self.model.get_text_outputs(input_ids)
        img_features = self.model.get_image_features_XAttnT2I(text_last_hidden, img)
        return nn.functional.normalize(text_features + img_features)

    def target_forward(self, img):
        img = img.to(self.di.device)
        target_features = self.model.get_image_features(img)
        return nn.functional.normalize(target_features)


class FirstHiddenXAttnT2I(DualEncoderModel):
    def __init__(self, path="clip-vit-large-patch14"):
        # load CLIP model & CLIP tokenizer
        super().__init__()
        self.model = CLIPModel.from_pretrained(path, local_files_only=True)
        self.model.get_text_outputs = MethodType(get_text_outputs, self.model)
        self.model.get_image_features_XAttnT2I = MethodType(get_image_features_XAttnT2I, self.model)
        self.model.vision_model.CLIP_vit_forward_XAttnT2I = MethodType(CLIP_vit_forward_XAttnT2I, self.model.vision_model)
        self.model.vision_model.encoder.CLIP_encoder_forward_XAttnT2I = MethodType(CLIP_encoder_forward_XAttnT2I_FirstHidden, self.model.vision_model.encoder)
        self.model.vision_model.encoder.xattn_layer = nn.MultiheadAttention(embed_dim=1024, num_heads=1, kdim=768, vdim=768, batch_first=True)
        self.model.vision_model.encoder.layer_norm = nn.LayerNorm(normalized_shape=(257, 1024))
        self.tokenizer = CLIPTokenizer.from_pretrained(path, local_files_only=True)

    def query_forward(self, img, text):
        input_ids = self.tokenizer(text, padding='max_length', return_tensors='pt').input_ids.to(self.di.device)
        img = img.to(self.di.device)
        text_features, text_last_hidden = self.model.get_text_outputs(input_ids)
        img_features = self.model.get_image_features_XAttnT2I(text_last_hidden, img)
        return nn.functional.normalize(text_features + img_features)

    def target_forward(self, img):
        img = img.to(self.di.device)
        target_features = self.model.get_image_features(img)
        return nn.functional.normalize(target_features)


class MaskedFirstHiddenXAttnT2I(DualEncoderModel):
    def __init__(self, path="clip-vit-large-patch14"):
        # load CLIP model & CLIP tokenizer
        super().__init__()
        self.model = CLIPModel.from_pretrained(path, local_files_only=True)
        self.model.get_text_outputs = MethodType(get_text_outputs, self.model)
        self.model.get_image_features_XAttnT2I = MethodType(get_image_features_XAttnT2I, self.model)
        self.model.vision_model.CLIP_vit_forward_XAttnT2I = MethodType(CLIP_vit_forward_XAttnT2I, self.model.vision_model)
        self.model.vision_model.encoder.CLIP_encoder_forward_XAttnT2I = MethodType(CLIP_encoder_forward_XAttnT2I_FirstHidden, self.model.vision_model.encoder)
        self.model.vision_model.encoder.xattn_layer = nn.MultiheadAttention(embed_dim=1024, num_heads=1, kdim=768, vdim=768, batch_first=True)
        self.model.vision_model.encoder.layer_norm = nn.LayerNorm(normalized_shape=(257, 1024))
        self.tokenizer = CLIPTokenizer.from_pretrained(path, local_files_only=True)

    def query_forward(self, img, text):
        input_ids = self.tokenizer(text, padding='max_length', return_tensors='pt').input_ids.to(self.di.device)
        eos_token_id = self.model.config.text_config.eos_token_id
        eos_positions = (input_ids == eos_token_id).int().argmax(dim=-1).to(self.di.device)
        img = img.to(self.di.device)
        text_features, text_last_hidden = self.model.get_text_outputs(input_ids)
        
        img_features = []
        for idx, p in enumerate(eos_positions):
            text_last_hidden_capped = text_last_hidden[idx, :p, :].unsqueeze(0)
            image = img[idx].unsqueeze(0)
            img_features.append(self.model.get_image_features_XAttnT2I(text_last_hidden_capped, image))
        img_features = torch.cat(img_features, dim=0)
        return nn.functional.normalize(text_features + img_features)

    def target_forward(self, img):
        img = img.to(self.di.device)
        target_features = self.model.get_image_features(img)
        return nn.functional.normalize(target_features)


""" Deprecated """
class CrossAttention(nn.Module):
    def __init__(self, src_dim, tgt_dim):
        # src_dim=1024, tgt_dim=768
        super().__init__()
        self.q_proj = nn.Linear(src_dim, tgt_dim)
        self.k_proj = nn.Linear(tgt_dim, tgt_dim)
        self.v_proj = nn.Linear(tgt_dim, tgt_dim)
        self.softmax = nn.Softmax(dim = -1)
        self.final_proj = nn.Linear(tgt_dim, src_dim)
    
    def forward(self, query, target):
        q = self.q_proj(query)
        k = self.k_proj(target).transpose(1, 2)
        v = self.v_proj(target)
        scores = torch.bmm(q, k)
        scores = self.softmax(scores)
        context = torch.bmm(scores, v)
        attn_out = self.final_proj(context)
        return attn_out
    

def get_image_outputs(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    vision_outputs = self.vision_model(
        pixel_values=pixel_values,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    vision_last_hidden = vision_outputs[0]
    image_features = self.text_projection(vision_outputs[1])

    return image_features, vision_last_hidden


def get_text_outputs(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    text_outputs = self.text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    text_last_hidden = text_outputs[0]
    text_features = self.text_projection(text_outputs[1])

    return text_features, text_last_hidden


def get_image_features_XAttnT2I(
        self,
        text_last_hidden,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    vision_outputs = self.vision_model.CLIP_vit_forward_XAttnT2I(
        text_last_hidden=text_last_hidden,
        pixel_values=pixel_values,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    pooled_output = vision_outputs[1]  # pooled_output
    image_features = self.visual_projection(pooled_output)

    return image_features


def CLIP_vit_forward_XAttnT2I(
        self,
        text_last_hidden,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if pixel_values is None:
        raise ValueError("You have to specify pixel_values")

    hidden_states = self.embeddings(pixel_values)
    hidden_states = self.pre_layrnorm(hidden_states)

    encoder_outputs = self.encoder.CLIP_encoder_forward_XAttnT2I(
        text_last_hidden=text_last_hidden,
        inputs_embeds=hidden_states,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    pooled_output = last_hidden_state[:, 0, :]
    pooled_output = self.post_layernorm(pooled_output)

    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )


def CLIP_encoder_forward_XAttnT2I_LastHidden(
        self,
        text_last_hidden,
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
    
    # Modification
    hidden_states += self.xattn_layer(hidden_states, text_last_hidden, text_last_hidden)[0]
    hidden_states = self.layer_norm(hidden_states)

    if output_hidden_states:
        encoder_states = encoder_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
    return BaseModelOutput(
        last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
    )


def CLIP_encoder_forward_XAttnT2I_FirstHidden(
        self,
        text_last_hidden,
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
            # Modification
            if idx == 0:
                hidden_states += self.xattn_layer(hidden_states, text_last_hidden, text_last_hidden)[0]
                hidden_states = self.layer_norm(hidden_states)

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


if __name__ == '__main__':
    pixel_values = torch.randn(1, 3, 224, 224).to("cuda")
    text = "Hello There"
    model = CLIPModel.from_pretrained("clip-vit-large-patch14", local_files_only=True).to("cuda")
    tokenizer = CLIPTokenizer.from_pretrained("clip-vit-large-patch14", local_files_only=True)
    input_ids = tokenizer(text, padding='max_length', return_tensors='pt').input_ids.to("cuda")

    model.get_image_outputs = MethodType(get_image_outputs, model)
    last_image_hidden = model.get_image_outputs(pixel_values)[1]

    model.get_text_outputs = MethodType(get_text_outputs, model)
    last_text_hidden = model.get_text_outputs(input_ids)[1]

    print(last_image_hidden.shape, last_text_hidden.shape)
