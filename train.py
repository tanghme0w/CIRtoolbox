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
import copy

logger = logging.Logger("main")

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

def main():
    # fix seed 
    fix_seed(42)
    
    # read config
    config = yaml.safe_load(open('config.yaml'))
    # cli args can override config
    args = parse_args()
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

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
    if config['ema']:
        model_params = list(model.parameters())
        ema_params = copy.deepcopy(model_params)

    # dataset & dataloader
    ds_name = config['dataset']
    try:
        train_loader, val_loader, target_loader = build_data(ds_name, config['batch_size'], model.preprocess)
    except AttributeError:
        train_loader, val_loader, target_loader = build_data(ds_name, config['batch_size'])

    # optimizer and scaler
    optimizer = AdamW(
        [{'params': model.parameters(), 'lr': config['lr'],
          'betas': (config['beta1'], config['beta2']), 'eps': config['eps']}])
    scaler = torch.cuda.amp.GradScaler()
    
    # loss function
    loss_func = nn.CrossEntropyLoss()

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
        for epoch in range(config['epochs']):
            # train
            model.train()
            epoch_loss = []
            train_bar = tqdm(train_loader, desc=f"[Train] epoch {epoch}")
            
            for idx, (ref_name, ref_img, captions, target_img) in enumerate(train_bar):

                optimizer.zero_grad()
                # import torch.amp
                with torch.cuda.amp.autocast():
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
                # scheduler.step()
                
                epoch_loss.append(loss)

                # EMA
                if config['ema']:
                    update_ema(ema_params, model_params, rate=config['ema'])

                # log step loss
                if config['enable_wandb']:
                    wandb.log({"loss": loss})

            # log epoch loss
            epoch_loss = torch.mean(torch.stack(epoch_loss, dim=0))
            loss_history.append(epoch_loss)
            print(f"epoch {epoch} loss: {epoch_loss}")

            """ Validate Epoch """
            if epoch == 0 or epoch >= 8:
                # initialize
                model.eval()
                metrics = [10, 50]
                recall = dict()

                # switch to ema params
                if config['ema']:
                    model_state_dict = copy.deepcopy(model.state_dict())
                    ema_state_dict = copy.deepcopy(model.state_dict())
                    for i, (name, _value) in enumerate(model.named_parameters()):
                        assert name in ema_state_dict
                        ema_state_dict[name] = ema_params[i]
                    print("Switch to ema")
                    model.load_state_dict(ema_state_dict)

                with torch.no_grad():
                    with torch.cuda.amp.autocast():
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
            
                    # back to no ema
                    if config['ema']:
                        print("Switch back from ema")
                        model.load_state_dict(model_state_dict)

    except Exception as e:
        print(traceback.format_exc())
        print(e)
        os.rmdir(os.path.join('saved', f'{start_time}_{ds_name}'))
    print(f"best_epoch: epoch{best_epoch}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
