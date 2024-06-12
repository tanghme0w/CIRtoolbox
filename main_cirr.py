import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import torch.nn as nn
import numpy as np
from transformers import CLIPModel, CLIPTokenizer
from data import FashionIQDataset, CIRRDataset
from tqdm import tqdm
from datetime import datetime
import os
import traceback

from preprocess import targetpad_transform
from params import parse_args
from fiq_caption_utils import combine_captions

import wandb


def main():
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # parse args
    args = parse_args()

    # setup wandb logging
    wandb.login()
    run = wandb.init(
        project="CIR-clipsum-cirr",
        config={
            "learning_rate": args.lr,
            "source_data": args.source_data
        }
    )

    # load CLIP model & CLIP tokenizer
    clip_model = CLIPModel.from_pretrained("clip-vit-large-patch14", local_files_only=True).to(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained("clip-vit-large-patch14", local_files_only=True)

    # create dataset
    train_query_dataset = CIRRDataset(
        mode='query', 
        split='train', 
        path='./cirr',
        preprocess=targetpad_transform()
    )

    val_query_dataset = CIRRDataset(
        mode='query', 
        split='val', 
        path='./cirr',
        preprocess=targetpad_transform()
    )

    target_dataset = CIRRDataset(
        mode='target', 
        path='./cirr',
        preprocess=targetpad_transform()
    )
    
    # create dataloaders
    train_loader = DataLoader(
        dataset=train_query_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=val_query_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        shuffle=False
    )

    relative_train_loader = DataLoader(dataset=train_query_dataset, batch_size=args.batch_size,
                                    num_workers=0, pin_memory=False,
                                    drop_last=True, shuffle=True)

    target_loader = DataLoader(
        dataset=target_dataset,
        batch_size=128
    )

    # freeze parameters
    # trainable_names = [
    #     "vision_model",
    #     "visual_projection"
    #     "text_model",
    #     "text_projection"
    # ]
    # for name, param in clip_model.named_parameters():
    #     if any(trainable_name in name for trainable_name in trainable_names):
    #         param.requires_grad = False
    
    # define optimizer, scaler, and scheduler
    optimizer = AdamW(
        [{'params': filter(lambda p: p.requires_grad, clip_model.parameters()), 'lr': args.lr,
          'betas': (0.9, 0.999), 'eps': 1e-7}])
    scaler = torch.cuda.amp.GradScaler()
    # define loss function
    loss_func = nn.CrossEntropyLoss()

    # info preparation
    start_time = datetime.now().strftime("%Y%m%d-%H-%M-%S")
    save_dir = os.path.join('saved', f'{start_time}_{args.source_data}')
    os.makedirs(save_dir)

    # run training script
    best_epoch = None
    best_eval_mean = 0
    patience_count = 0
    try:
        loss_history = []
        for epoch in range(args.epochs):
            # train
            clip_model.train()
            epoch_loss = []
            train_bar = tqdm(relative_train_loader, desc=f"[Train] epoch {epoch}")
            
            for idx, (ref_name, ref_img, captions, target_img) in enumerate(train_bar):

                optimizer.zero_grad()

                # process captions
                combined_captions = list(captions)

                with torch.cuda.amp.autocast():
                    # get target features
                    target_img = target_img.to(device)
                    target_features = clip_model.get_image_features(target_img)
                    
                    # get query features
                    text_input = clip_tokenizer(combined_captions, padding="max_length", return_tensors='pt').input_ids
                    text_input = text_input.to(device)
                    text_features = clip_model.get_text_features(text_input)

                    ref_img = ref_img.to(device)
                    ref_img_features = clip_model.get_image_features(ref_img)
                    
                    composed_features = text_features + ref_img_features

                    # compute loss
                    target_features = torch.nn.functional.normalize(target_features, dim=-1)
                    composed_features = torch.nn.functional.normalize(composed_features, dim=-1)
                    logits = 100 * composed_features @ target_features.T
                    labels = torch.arange(ref_img.shape[0], dtype=torch.long, device=device)
                    loss = loss_func(logits, labels)

                # optimizer step
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss.append(loss)

                # log step loss
                wandb.log({"loss": loss})

            # log epoch loss
            epoch_loss = torch.mean(torch.stack(epoch_loss, dim=0))
            loss_history.append(epoch_loss)
            print(f"epoch {epoch} loss: {epoch_loss}")

            # validate
            clip_model.eval()
            metrics = [1, 5, 10, 50, 100]
            recall = dict()
            with torch.no_grad():
                # get all target feature from validation set
                target_bar = tqdm(target_loader, desc="[Validate] Target Feature")
                all_target_names = []
                all_target_features = []
                for img_name, img in target_bar:
                    all_target_names += img_name
                    all_target_features.append(clip_model.get_image_features(img.to(device)))
                all_target_features = torch.nn.functional.normalize(torch.cat(all_target_features))
                # get all query feature for validation
                val_bar = tqdm(val_loader, desc="[Validate] Query Feature")
                for i in metrics:
                    recall[i] = 0
                count = 0
                for ref_name, ref_img, captions, target_name in val_bar:
                    # infer ref_img features
                    ref_img_features = clip_model.get_image_features(ref_img.to(device))
                    flattened_captions: list = np.array(captions).T.flatten().tolist()
                    input_captions = list(captions)
                    input_ids = clip_tokenizer(input_captions, padding='max_length', return_tensors='pt').input_ids.to(device)
                    text_features = clip_model.get_text_features(input_ids)
                    composed_features = ref_img_features + text_features
                    target_idx = [all_target_names.index(name) for name in target_name]
                    # compute sorted index matrix
                    composed_features = torch.nn.functional.normalize(composed_features)
                    all_target_features = torch.nn.functional.normalize(all_target_features)
                    distances = 1 - composed_features @ all_target_features.T
                    sorted_indices = torch.argsort(distances, dim=-1).cpu()
                    index_mask = torch.Tensor(target_idx).view(len(target_idx), 1).repeat(1, sorted_indices.shape[-1])
                    labels = sorted_indices.eq(index_mask).float()
                    for metric in metrics:
                        recall[metric] += torch.sum(labels[:, :metric])
                    count += len(ref_img)
                for i in metrics:
                    recall[i] /= count
                print([f"R@{i}: {recall[i]}" for i in metrics])
            
            eval_mean = sum([recall[i] for i in metrics]) / len(metrics)
            if eval_mean > best_eval_mean:
                patience_count = 0
                best_eval_mean = eval_mean
                best_epoch = epoch
                # save checkpoint
                checkpoint = {
                    "epoch": epoch,
                    "model_sd": clip_model.state_dict(),
                    "optimizer_sd": optimizer.state_dict()
                }
                torch.save(checkpoint, os.path.join(save_dir, f"epoch{str(epoch).zfill(3)}.pt"))
            else:
                patience_count += 1
                if patience_count > args.patience:
                    break

    except Exception as e:
        print(traceback.format_exc())
        print(e)
        os.rmdir(os.path.join('saved', f'{start_time}_{args.source_data}'))
    print(f"best_epoch: epoch{best_epoch}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
