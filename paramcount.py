from transformers import CLIPModel
from eva_clip import create_model_and_transforms, get_tokenizer


model_name = "EVA02-CLIP-L-14-336" 
pretrained = "../eva-clip/EVA02_CLIP_L_336_psz14_s6B.pt" # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"

model, _, preprocess = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)

# model = CLIPModel.from_pretrained("clip-vit-large-patch14", local_files_only=True)

print(sum(p.numel() for p in model.parameters()))
