from time import time
from datetime import datetime
import os

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from CIRtoolbox.clip_ca import CLIPModel
from transformers import CLIPTokenizer
from CIRtoolbox.variations.visinv import VisualInversion, CrossAttention
from tqdm import tqdm


def calculate_loss(
        clip_model: CLIPModel,
        clip_tokenizer: CLIPTokenizer,
        loss_func: torch.nn.Module,
        ref_img,
        text,
        tgt_img,
        args
):
    # get source embedding
    tokenized_text = clip_tokenizer(text, padding=True, return_tensors="pt")
    if args.gpu is not None:
        tokenized_text.to(f"cuda:{args.gpu}")
    text_feature = clip_model.get_text_features(**tokenized_text)
    image_feature = clip_model.get_image_features(ref_img)
    composed_feature = F.normalize(text_feature + image_feature, dim=-1)
    # get target embedding
    target_feature = clip_model.get_image_features(tgt_img)
    # calculate loss and return
    composed_feature = torch.nn.functional.normalize(composed_feature, dim=-1)
    target_feature = torch.nn.functional.normalize(target_feature, dim=1)   # normalization
    scores = composed_feature @ target_feature.T / args.temperature
    labels = torch.arange(composed_feature.shape[0]).to(composed_feature.device)
    loss = loss_func(scores, labels)
    return loss


def train(
        save_file,
        clip_model,
        clip_tokenizer,
        fm_model,
        visinv_attn,
        ln,
        dataloader,
        epoch,
        optimizer,
        scaler,
        scheduler,
        args
):
    clip_model.eval()
    # get batches per epoch
    num_batches_per_epoch = len(dataloader)
    # track time
    time_track = time()
    # track loss
    loss_func = torch.nn.CrossEntropyLoss()
    epoch_loss = torch.tensor(0.)
    # training pipeline
    for i, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch}: ")):
        global_step_num = num_batches_per_epoch * epoch + i
        scheduler(global_step_num)
        optimizer.zero_grad()

        ref_images, target_images, _, _, answer_paths, _, captions = batch

        data_identifier = -1
        if args.gpu is not None:
            ref_images = ref_images.cuda(args.gpu, non_blocking=True)   # todo what does non_blocking mean
            target_images = target_images.cuda(args.gpu, non_blocking=True)

        # with automatic mixed precision.
        if args.precision == "amp":
            with autocast():
                total_loss = calculate_loss(
                    clip_model=clip_model,
                    clip_tokenizer=clip_tokenizer,
                    fm_model=fm_model,
                    visinv_attn=visinv_attn,
                    ln=ln,
                    loss_func=loss_func,
                    ref_img=ref_images,
                    text=captions,
                    tgt_img=target_images,
                    args=args
                )
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            total_loss = calculate_loss(
                clip_model=clip_model,
                clip_tokenizer=clip_tokenizer,
                loss_func=loss_func,
                ref_img=ref_images,
                text=captions,
                tgt_img=target_images,
                args=args
            )
            total_loss.backward()
            optimizer.step()
        # track loss
        epoch_loss += total_loss.cpu()

    epoch_loss = epoch_loss / num_batches_per_epoch
    # save model
    checkpoint = {
        'clip_state_dict': clip_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, save_file)
    epoch_time = time() - time_track
    return epoch_time, epoch_loss.item()
