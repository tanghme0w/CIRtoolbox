import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import torch.nn as nn
import numpy as np
from transformers import CLIPModel, CLIPTokenizer
from data import FashionIQDataset
from tqdm import tqdm
from datetime import datetime
import os
import traceback

from preprocess import targetpad_transform
from params import parse_args
from fiq_caption_utils import combine_captions


def main():
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # parse args
    args = parse_args()

    # load CLIP model & CLIP tokenizer
    clip_model = CLIPModel.from_pretrained("clip-vit-large-patch14", local_files_only=True).to(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained("clip-vit-large-patch14", local_files_only=True)

    # learnable MLP gate
    hidden_dim = 768
    combiner_fc = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim),
                                         nn.ReLU()).to(device)
    dropout = nn.Dropout(0.5).to(device)
    scaler_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(hidden_dim, 1),
                                       nn.Sigmoid()).to(device)
    

    # create dataset
    train_query_dataset = FashionIQDataset(
        mode='query', 
        clothtype=args.source_data, 
        split='train', 
        path='./fashion-iq',
        preprocess=targetpad_transform()
    )

    val_query_dataset = FashionIQDataset(
        mode='query', 
        clothtype=args.source_data, 
        split='val', 
        path='./fashion-iq',
        preprocess=targetpad_transform()
    )

    target_dataset = FashionIQDataset(
        mode='target', 
        clothtype=args.source_data,
        path='./fashion-iq',
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
    #     "visual_projection",
    #     "text_model",
    #     "text_projection"
    # ]
    # for name, param in clip_model.named_parameters():
    #     # if any(trainable_name in name for trainable_name in trainable_names):
    #     param.requires_grad = True
    
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
            
            for idx, (ref_img, captions, target_img) in enumerate(train_bar):
                # import ipdb; ipdb.set_trace()

                optimizer.zero_grad()

                # process captions
                flattened_captions: list = np.array(captions).T.flatten().tolist()
                combined_captions = combine_captions(flattened_captions)

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
                    
                    # gating combination
                    # import ipdb; ipdb.set_trace()
                    composed_features = combiner_fc(torch.cat([text_features, ref_img_features], dim=-1))
                    dynamic_scaler = scaler_fc(dropout(composed_features))
                    composed_features = dynamic_scaler * text_features + (1 - dynamic_scaler) * ref_img_features

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
                # hit1 = 0
                # hit5 = 0
                # hit10 = 0
                # hit50 = 0
                # hit100 = 0
                count = 0
                for ref_img, captions, target_name in val_bar:
                    # infer ref_img features
                    ref_img_features = clip_model.get_image_features(ref_img.to(device))
                    flattened_captions: list = np.array(captions).T.flatten().tolist()
                    input_captions = [
                        f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
                        i in range(0, len(flattened_captions), 2)]
                    input_ids = clip_tokenizer(input_captions, padding='max_length', return_tensors='pt').input_ids.to(device)
                    text_features = clip_model.get_text_features(input_ids)
                    # composed features with gating
                    composed_features = combiner_fc(torch.cat([text_features, ref_img_features], dim=-1))
                    dynamic_scaler = scaler_fc(dropout(composed_features))
                    composed_features = dynamic_scaler * text_features + (1 - dynamic_scaler) * ref_img_features
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
                    # hit1 += torch.sum(labels[:, 0:1])
                    # hit5 += torch.sum(labels[:, 0:5])
                    # hit10 += torch.sum(labels[:, 0:10])
                    # hit50 += torch.sum(labels[:, 0:50])
                    # hit100 += torch.sum(labels[:, 0:100])
                    count += len(ref_img)
                for i in metrics:
                    recall[i] /= count
                print([f"R@{i}: {recall[i]}" for i in metrics])
                # hit1 = hit1 / count
                # hit5 = hit5 / count
                # hit10 = hit10 / count
                # hit50 = hit50 / count
                # hit100 = hit100 / count
                
                # print(f"R@1: {recall[1]}, R@5: {hit5}, R@10: {hit10}, R@50: {hit50}, R@100: {hit100}")
            
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
