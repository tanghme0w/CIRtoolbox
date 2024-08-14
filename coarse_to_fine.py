import torch
from torch.optim import AdamW
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
import os
import traceback

from data import build_data
from params import parse_args
from model import *
import yaml

import wandb
import logging

logger = logging.Logger("main")

def main():
    # read config
    config = yaml.safe_load(open('config.yaml'))
    # cli args can override config
    args = parse_args()
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    # set device
    if not torch.cuda.is_available():
        logger.log("cuda device not available, device settings will be neglected.")
        device = "cpu"
    else:
        device = config['device']

    # setup wandb logging
    if config['enable_wandb']:
        wandb.login()
        run = wandb.init(
            project="CIR-clipsum",
            config={
                "learning_rate": config['lr'],
                "dataset": config['dataset']
            }
        )

    # model
    model = FineGrainModel().to(device)
    # """ Freeze transformer """
    # for name, param in model.named_parameters():
    #     if not name.startswith("model."):
    #         param.requires_grad = False

    # dataset & dataloader
    ds_name = config['dataset']
    try:
        train_loader, val_loader, target_loader = build_data(ds_name, config['batch_size'], model.preprocess)
    except AttributeError:
        train_loader, val_loader, target_loader = build_data(ds_name, config['batch_size'])

    # optimizer, scaler, and scheduler
    optimizer = AdamW(
        [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': config['lr'],
          'betas': (config['beta1'], config['beta2']), 'eps': config['eps']}])
    scaler = torch.amp.GradScaler('cuda')
    
    # loss function
    loss_func = nn.CrossEntropyLoss()
    stage2_loss = nn.CrossEntropyLoss()

    # info preparation
    start_time = datetime.now().strftime("%Y%m%d-%H-%M-%S")
    save_dir = os.path.join('saved', f'{start_time}_{ds_name}')
    os.makedirs(save_dir)

    # run training script
    best_epoch = None
    best_eval_mean = 0
    patience_count = 0
    try:
        loss_history = []
        """ 
        Stage 1: Coarse-grained training
        """
        for epoch in range(1):
            # train
            model.train()
            epoch_loss = []
            train_bar = tqdm(train_loader, desc=f"[Stage 1 Train] epoch {epoch}")
            
            for idx, (ref_name, ref_img, captions, target_img) in enumerate(train_bar):

                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    # forward
                    query_emb = model.query_forward(ref_img, captions)
                    tgt_emb = model.target_forward(target_img)

                    # compute loss
                    logits = 100 * query_emb @ tgt_emb.T
                    labels = torch.arange(ref_img.shape[0], dtype=torch.long, device=device)
                    loss = loss_func(logits, labels)

                # optimizer step
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss.append(loss)

                # log step loss
                if config['enable_wandb']:
                    wandb.log({"loss": loss})

            # log epoch loss
            epoch_loss = torch.mean(torch.stack(epoch_loss, dim=0))
            loss_history.append(epoch_loss)
            print(f"epoch {epoch} loss: {epoch_loss}")

            # validate epoch
            model.eval()
            metrics = [10, 50]
            recall = dict()
            with torch.no_grad():
                # get all target feature from validation set
                target_bar = tqdm(target_loader, desc="[Validate] Target Feature")
                all_target_names = []
                all_target_features = []
                for img_name, img in target_bar:
                    all_target_names += img_name
                    all_target_features.append(model.target_forward(img))
                all_target_features = torch.cat(all_target_features)
                # get all query feature for validation
                val_bar = tqdm(val_loader, desc="[Validate] Query Feature")
                for i in metrics:
                    recall[i] = 0
                count = 0
                for ref_name, ref_img, captions, target_name in val_bar:
                    # infer query features
                    query_emb = model.query_forward(ref_img, captions)
                    target_idx = [all_target_names.index(name) for name in target_name]
                    # compute sorted index matrix
                    distances = 1 - query_emb @ all_target_features.T
                    sorted_indices = torch.argsort(distances, dim=-1).cpu()
                    index_mask = torch.Tensor(target_idx).view(len(target_idx), 1).repeat(1, sorted_indices.shape[-1])
                    labels = sorted_indices.eq(index_mask).float()
                    for metric in metrics:
                        recall[metric] += torch.sum(labels[:, :metric])
                    count += len(ref_img)
                for i in metrics:
                    recall[i] /= count
                    recall[i] *= 100
                print([f"R@{i}: {recall[i]}" for i in metrics])

            eval_mean = sum([recall[i] for i in metrics]) / len(metrics)
            if eval_mean > best_eval_mean:
                patience_count = 0
                best_eval_mean = eval_mean
                best_epoch = epoch
                # save checkpoint
                checkpoint = {
                    "epoch": epoch,
                    "model_sd": model.state_dict(),
                    "optimizer_sd": optimizer.state_dict()
                }
                torch.save(checkpoint, os.path.join(save_dir, f"epoch{str(epoch).zfill(3)}.pt"))
            else:
                patience_count += 1
                if patience_count > config['patience']:
                    break
        
        """ 
        Stage 2: Fine-grained training
        """

        """ Freeze image model """
        for name, param in model.named_parameters():
            if name.startswith("model."):
                param.requires_grad = False

        """ optimizer, scaler, and scheduler """
        optimizer = AdamW(
            [{'params': [ p for p in model.parameters() if p.requires_grad], 'lr': config['lr'],
            'betas': (config['beta1'], config['beta2']), 'eps': config['eps']}])
        scaler = torch.amp.GradScaler('cuda')

        for epoch in range(config['epochs']):
            # train
            model.train()
            epoch_loss = []
            train_bar = tqdm(train_loader, desc=f"[Stage 2 Train] epoch {epoch}")
            
            for idx, (ref_name, ref_img, captions, target_img) in enumerate(train_bar):

                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    # forward
                    _, query_fg = model.fine_grained_query_forward(ref_img, captions)
                    tgt_fg = model.fine_grained_target_forward(target_img)

                    # compute loss
                    loss = 0
                    bs = query_fg.shape[0]
                    for qfg, tfg in zip(query_fg, tgt_fg):         
                        logits = 100 * qfg @ tfg.T
                        labels = torch.arange(qfg.shape[0], dtype=torch.long, device=device)
                        loss += stage2_loss(logits, labels)
                    loss /= bs

                # optimizer step
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss.append(loss)

                # log step loss
                if config['enable_wandb']:
                    wandb.log({"loss": loss})

            # log epoch loss
            epoch_loss = torch.mean(torch.stack(epoch_loss, dim=0))
            loss_history.append(epoch_loss)
            print(f"epoch {epoch} loss: {epoch_loss}")

            # validate epoch
            model.eval()
            metrics = [10, 50]
            recall = dict()
            with torch.no_grad():
                # get all target feature from validation set
                target_bar = tqdm(target_loader, desc="[Validate] Target Feature")
                all_target_names = []
                all_target_features = []
                all_target_fg_feat = []
                for img_name, img in target_bar:
                    all_target_names += img_name
                    all_target_features.append(model.target_forward(img))
                    all_target_fg_feat.append(model.fine_grained_target_forward(img))
                all_target_features = torch.cat(all_target_features)
                all_target_fg_feat = torch.cat(all_target_fg_feat, dim=0)
                # get all query feature for validation
                val_bar = tqdm(val_loader, desc="[Validate] Query Feature")
                for i in metrics:
                    recall[i] = 0
                count = 0
                for ref_name, ref_img, captions, target_name in val_bar:
                    # infer query features
                    query_emb = model.query_forward(ref_img, captions)
                    query_fg_emb = model.fine_grained_query_forward(ref_img, captions)[1]
                    target_idx = [all_target_names.index(name) for name in target_name]
                    # compute sorted index matrix
                    similarity_c = query_emb @ all_target_features.T
                    similarity_f = []
                    for qfg in query_fg_emb:
                        local_sim = []
                        for tfg in all_target_fg_feat:
                            local_sim.append((torch.sum(torch.diag(qfg @ tfg.T))/qfg.shape[0]).unsqueeze(0))
                        similarity_f.append(torch.cat(local_sim).unsqueeze(0))
                    similarity_f = torch.cat(similarity_f, dim=0)
                    similarity = 0.5 * similarity_c + 0.5 * similarity_f
                    sorted_indices = torch.argsort(similarity, dim=-1, descending=True).cpu()
                    index_mask = torch.Tensor(target_idx).view(len(target_idx), 1).repeat(1, sorted_indices.shape[-1])
                    labels = sorted_indices.eq(index_mask).float()
                    for metric in metrics:
                        recall[metric] += torch.sum(labels[:, :metric])
                    count += len(ref_img)
                for i in metrics:
                    recall[i] /= count
                    recall[i] *= 100
                print([f"R@{i}: {recall[i]}" for i in metrics])

            eval_mean = sum([recall[i] for i in metrics]) / len(metrics)
            if eval_mean > best_eval_mean:
                patience_count = 0
                best_eval_mean = eval_mean
                best_epoch = epoch
                # save checkpoint
                checkpoint = {
                    "epoch": epoch,
                    "model_sd": model.state_dict(),
                    "optimizer_sd": optimizer.state_dict()
                }
                torch.save(checkpoint, os.path.join(save_dir, f"epoch{str(epoch).zfill(3)}.pt"))
            else:
                patience_count += 1
                if patience_count > config['patience']:
                    break
            

    except Exception as e:
        print(traceback.format_exc())
        print(e)
        os.rmdir(os.path.join('saved', f'{start_time}_{ds_name}'))
    print(f"best_epoch: epoch{best_epoch}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()