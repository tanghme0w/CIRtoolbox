import torch
from torch.optim import AdamW
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
import os
import traceback

from data import build_data, build_ddp_data
from params import parse_args
from model import *
import yaml

import wandb
import logging

logger = logging.Logger("main")

# for distribued training
import torch.distributed as dist
import builtins

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (dist.get_world_size() > 8)
        if is_master or force:
            now = datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def init_distributed_mode(config):
    if config['mpi']:
        config['rank'] = int(os.environ['OMPI_COMM_WORLD_RANK'])
        config['world_size'] = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        config['gpu'] = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        config['dist_url'] = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(config['gpu'])
        os.environ['RANK'] = str(config['rank'])
        os.environ['WORLD_SIZE'] = str(config['world_size'])
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        config['rank'] = int(os.environ["RANK"])
        config['world_size'] = int(os.environ['WORLD_SIZE'])
        config['gpu'] = int(os.environ['LOCAL_RANK'])
        config['dist_url'] = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
    elif 'SLURM_PROCID' in os.environ:
        config['rank'] = int(os.environ['SLURM_PROCID'])
        config['gpu'] = config['rank'] % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        config['ddp'] = False
        return

    config['ddp'] = True

    torch.cuda.set_device(config['gpu'])
    config['dist_backend'] = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        config['rank'], config['dist_url'], config['gpu']), flush=True)
    torch.distributed.init_process_group(backend=config['dist_backend'], init_method=config['dist_url'],
                                         world_size=config['world_size'], rank=config['rank'])
    torch.distributed.barrier()
    setup_for_distributed(config['rank'] == 0)


def main():
    # read config
    config = yaml.safe_load(open('config.yaml'))
    # cli args can override config
    args = parse_args()
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    # init_distributed_mode
    init_distributed_mode(config)

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    model = LowLevelFusion().to(device)
    model_without_ddp = model
    if config['ddp']:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config['gpu']])
        model_without_ddp = model.module

    # dataset & dataloader
    ds_name = config['dataset']
    if config['ddp']:
        train_loader, val_loader, target_loader = build_ddp_data(
            name=ds_name, 
            bs=config['batch_size'], 
            rank=dist.get_rank(),
            world_size=dist.get_world_size(), 
            preprocess=model_without_ddp.preprocess
            )
    else:
        train_loader, val_loader, target_loader = build_data(
            name=ds_name, 
            bs=config['batch_size'], 
            preprocess=model_without_ddp.preprocess
            )

    # optimizer and scaler
    if config['ddp']:
        config['lr'] *= config['world_size']
    print(f"learning rate: {config['lr']}")
    optimizer = AdamW(
        [{'params': model.parameters(), 'lr': config['lr'],
          'betas': (config['beta1'], config['beta2']), 'eps': config['eps']}])
    scaler = torch.amp.GradScaler('cuda')

    # loss function
    loss_func = nn.CrossEntropyLoss()

    # info preparation
    if config['rank'] == 0:
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
            # set epoch if ddp
            if config['ddp']:
                train_loader.sampler.set_epoch(epoch)

            # train
            model.train()
            epoch_loss = []
            train_bar = tqdm(train_loader, desc=f"[Train] epoch {epoch}") if config['rank'] == 0 else train_loader
            
            for idx, (ref_name, ref_img, captions, target_img) in enumerate(train_bar):

                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    # forward
                    query_emb = model_without_ddp.query_forward(ref_img, captions)
                    tgt_emb = model_without_ddp.target_forward(target_img)

                    # compute loss
                    logits = 100 * query_emb @ tgt_emb.T
                    labels = torch.arange(ref_img.shape[0], dtype=torch.long, device=device)
                    loss = loss_func(logits, labels)

                # optimizer step
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # scheduler.step()

                epoch_loss.append(loss)

                # log step loss
                if config['enable_wandb']:
                    wandb.log({"loss": loss})

            # log epoch loss
            epoch_loss = torch.mean(torch.stack(epoch_loss, dim=0))
            loss_history.append(epoch_loss)
            print(f"epoch {epoch} loss: {epoch_loss}")

            """ Validate Epoch """
            # initialize
            if config['rank'] == 0:
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
                        all_target_features.append(model_without_ddp.target_forward(img))
                    all_target_features = torch.cat(all_target_features)
                    # get all query feature for validation
                    val_bar = tqdm(val_loader, desc="[Validate] Query Feature")
                    for i in metrics:
                        recall[i] = 0
                    count = 0
                    for ref_name, ref_img, captions, target_name in val_bar:
                        # infer query features
                        query_emb = model_without_ddp.query_forward(ref_img, captions)
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
            
            # synchronize
            torch.distributed.barrier()

    except Exception as e:
        print(traceback.format_exc())
        print(e)
        os.rmdir(os.path.join('saved', f'{start_time}_{ds_name}'))
    print(f"best_epoch: epoch{best_epoch}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
