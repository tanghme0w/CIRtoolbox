import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import torch.nn as nn

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from transformers import CLIPModel, CLIPTokenizer
from tqdm import tqdm
from datetime import datetime
import os
import traceback

from data_ddp import build_data
from params import parse_args
from model import *
import yaml

import wandb


def gather_tensor(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)

def gather_obj(data, world_size, rank):
    torch.cuda.set_device(rank)
    gathered_data = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_data, data)
    return gathered_data

def main():

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Start running on rank {rank}.")

    # read config
    config = yaml.safe_load(open('config.yaml'))
    # cli args can override config
    args = parse_args()
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    # setup wandb logging
    if rank == 0 and config['enable_wandb']:
        wandb.login()
        run = wandb.init(
            project="CIR-clipsum",
            config={
                "learning_rate": config['lr'],
                "dataset": config['dataset']
            }
        )

    # set device
    device_id = rank % torch.cuda.device_count()

    # model
    model = CLIPSumModel("clip-vit-large-patch14").to(device_id)
    model = DDP(model, device_ids=[device_id])

    # dataset & dataloader
    ds_name = config['dataset']
    train_loader, val_loader, target_loader = build_data(ds_name, config['batch_size'])

    # optimizer, scaler, and scheduler
    optimizer = AdamW(
        [{'params': model.parameters(), 'lr': config['lr'],
          'betas': (config['beta1'], config['beta2']), 'eps': config['eps']}])
    scaler = torch.cuda.amp.GradScaler()
    
    # loss function
    loss_func = nn.CrossEntropyLoss()

    # info preparation
    if rank == 0:
        start_time = datetime.now().strftime("%Y%m%d-%H-%M-%S")
        save_dir = os.path.join('saved', f'{start_time}_{ds_name}')
        os.makedirs(save_dir)

    # run training script
    best_epoch = None
    best_eval_mean = 0
    patience_count = 0
    try:
        loss_history = []
        for epoch in range(config['epochs']):
            # train
            model.train()
            epoch_loss = []
            train_bar = tqdm(train_loader, desc=f"[Train] epoch {epoch}", disable=rank!=0)
            
            for idx, (ref_name, ref_img, captions, target_img) in enumerate(train_bar):

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    # forward
                    query_emb = model.module.query_forward(ref_img, captions)
                    tgt_emb = model.module.target_forward(target_img)

                    # compute loss
                    logits = 100 * query_emb @ tgt_emb.T
                    labels = torch.arange(ref_img.shape[0], dtype=torch.long, device=device_id)
                    loss = loss_func(logits, labels)

                # optimizer step
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss.append(loss)

                # log step loss
                if rank == 0 and config['enable_wandb']:
                    wandb.log({"loss": loss})

            # log epoch loss
            if rank == 0:
                epoch_loss = torch.mean(torch.stack(epoch_loss, dim=0))
                loss_history.append(epoch_loss)
                print(f"epoch {epoch} loss: {epoch_loss}")

            # validate epoch
            model.eval()
            metrics = [1, 5, 10, 50, 100]
            recall = dict()
            with torch.no_grad():
                # get all target feature from validation set
                target_bar = tqdm(target_loader, desc="[Validate] Target Feature", disable=rank!=0)
                local_target_names = []
                local_target_features = []
                for img_name, img in target_bar:
                    local_target_names += img_name
                    local_target_features.append(model.module.target_forward(img))
                local_target_features = torch.cat(local_target_features)

                # Gather target features and names from all GPUs
                all_target_features = gather_tensor(local_target_features, world_size)
                all_target_names = gather_obj(local_target_names, world_size, rank)
                all_target_names = sum(all_target_names, [])

                # get all query feature for validation
                val_bar = tqdm(val_loader, desc="[Validate] Query Feature", disable=rank!=0)
                for i in metrics:
                    recall[i] = 0
                count = 0
                for ref_name, ref_img, captions, target_name in val_bar:
                    # infer query features
                    query_emb = model.module.query_forward(ref_img, captions)
                    target_idx = [all_target_names.index(name) for name in target_name]
                    # compute sorted index matrix
                    distances = 1 - query_emb @ all_target_features.T
                    sorted_indices = torch.argsort(distances, dim=-1).cpu()
                    index_mask = torch.Tensor(target_idx).view(len(target_idx), 1).repeat(1, sorted_indices.shape[-1])
                    labels = sorted_indices.eq(index_mask).float()
                    for metric in metrics:
                        recall[metric] += torch.sum(labels[:, :metric])
                    count += len(ref_img)

                # Gather hits and counts from all GPUs
                all_recall = dict()
                for i in metrics:
                    all_recall[i] = sum(gather_obj(recall[i], world_size, rank))
                all_count = sum(gather_obj(count, world_size, rank))

                if rank == 0:
                    # for i in metrics:
                    #     dist.all_reduce(torch.tensor(recall[i]).to(device_id), op=dist.ReduceOp.SUM)
                    # dist.all_reduce(torch.tensor(count).to(device_id), op=dist.ReduceOp.SUM)

                    for i in metrics:
                        all_recall[i] /= all_count
                    print([f"R@{i}: {all_recall[i]}" for i in metrics])

                    eval_mean = sum([all_recall[i] for i in metrics]) / len(metrics)
                    if rank == 0:
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
        if rank == 0:
            os.rmdir(os.path.join('saved', f'{start_time}_{ds_name}'))
        dist.destroy_process_group()
    print(f"best_epoch: epoch{best_epoch}")
    dist.destroy_process_group()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # world_size = int(os.environ.get('WORLD_SIZE', 1))
    # rank = int(os.environ.get('RANK', 0))

    main()
