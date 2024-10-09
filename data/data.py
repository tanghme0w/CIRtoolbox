from torch.utils.data import Dataset, DataLoader, default_collate
import torch
import os
import json
from PIL import Image
import numpy as np
from typing import List
import random
from .augmentation import data_augmentation

# FashionIQ Dataset
class FashionIQDataset(Dataset):
    def __init__(self, mode: str, clothtype: str, preprocess: callable, split: str = 'val', path: str='.'):
        super().__init__()

        # validate & store parameters
        assert mode in ['query', 'target']
        assert clothtype in ['dress', 'shirt', 'toptee']
        assert split in ['train', 'val', 'test']
        self.mode = mode
        self.preprocess = preprocess
        self.split = split
        self.image_path = os.path.join(path, 'images')

        # read metadata & split files
        metadata_file = os.path.join(path, f'captions/cap.{clothtype}.{split}.json')
        split_file = os.path.join(path, f'image_splits/split.{clothtype}.{split}.json')
        
        if mode == 'query':
            self.metadata = json.load(open(metadata_file))
            
        if mode == 'target':
            self.names = json.load(open(split_file))


    def __getitem__(self, index):
        if self.mode == 'target':
            image_name = self.names[index]
            image_file = os.path.join(self.image_path, f'{image_name}.png')
            image=Image.open(image_file)
            if self.split == 'train':
                image = data_augmentation(
                    image=image,
                    p=0.2,
                    mask_size=50,
                    mask_color=(0, 0, 0),
                    hgain=0.1,
                    sgain=0.3,
                    vgain=0.5,
                    angle=45,
                    expand=True,
                    scale_range=(0.5, 1.5),
                    mean=0,
                    std=25,
                )
            image = self.preprocess(image)
            return image_name, image

        elif self.mode == 'query':
            candidate_name = self.metadata[index]['candidate']
            target_name = self.metadata[index]['target']
            captions = self.metadata[index]['captions']
            candidate_image_file = os.path.join(self.image_path, f'{candidate_name}.png')

            if self.split == 'train':
                candidate_image = data_augmentation(
                    image=Image.open(candidate_image_file),
                    p=0.2,
                    mask_size=50,
                    mask_color=(0, 0, 0),
                    hgain=0.1,
                    sgain=0.3,
                    vgain=0.5,
                    angle=45,
                    expand=True,
                    scale_range=(0.5, 1.5),
                    mean=0,
                    std=25,
                )
                candidate_image = self.preprocess(candidate_image)

                target_image_file = os.path.join(self.image_path, f'{target_name}.png')
                target_image = data_augmentation(
                    image=Image.open(target_image_file),
                    p=0.2,
                    mask_size=50,
                    mask_color=(0, 0, 0),
                    hgain=0.1,
                    sgain=0.3,
                    vgain=0.5,
                    angle=45,
                    expand=True,
                    scale_range=(0.5, 1.5),
                    mean=0,
                    std=25,
                )
                target_image = self.preprocess(target_image)
                return candidate_name, candidate_image, captions, target_image

            else:
                candidate_image = self.preprocess(Image.open(candidate_image_file))
                return candidate_name, candidate_image, captions, target_name


    def __len__(self):
        if self.mode == 'target':
            return len(self.names)
        if self.mode == 'query':
            return len(self.metadata)
        

class CIRRDataset(Dataset):
    def __init__(self, mode, preprocess, split: str='val', path: str='.'):
        super().__init__()
        assert split in ['train', 'test1', 'val']
        assert mode in ['query', 'target']
        self.path = path
        self.split = split
        self.mode = mode
        self.preprocess = preprocess
        self.triplets = json.load(open(os.path.join(path, f'captions/cap.rc2.{self.split}.json')))
        self.namepath = json.load(open(os.path.join(path, f'image_splits/split.rc2.{split}.json')))
    

    def __getitem__(self, index):
        # index should not be batched
        if self.mode == 'target':
            image_name = list(self.namepath.keys())[index]
            image_rel_path = self.namepath[image_name]
            image_full_path = os.path.join(self.path, image_rel_path)
            image = self.preprocess(Image.open(image_full_path))
            return image_name, image

        if self.mode == 'query':
            ref_name = self.triplets[index]['reference']
            ref_path = os.path.join(self.path, self.namepath[ref_name])
            ref_img = self.preprocess(Image.open(ref_path))
            target_name = self.triplets[index]['target_hard']
            caption = self.triplets[index]['caption']
             
            if self.split == 'train':
                target_path = os.path.join(self.path, self.namepath[target_name])
                target_image = self.preprocess(Image.open(target_path))
                return ref_name, ref_img, caption, target_image
            else:
                return ref_name, ref_img, caption, target_name
            
    
    def __len__(self):
        if self.mode == 'target':
            return len(self.namepath)
        if self.mode == 'query':
            return len(self.triplets)


# reference: CLIP4CIR (CVPRW 2022)
def combine_captions(flattened_captions: List[str]):
    """
    Function which randomize the FashionIQ training captions in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1
    (d) cap2
    :param caption_pair: the list of caption to randomize, note that the length of such list is 2*batch_size since
     to each triplet are associated two captions
    :return: the randomized caption list (with length = batch_size)
    """
    captions = []
    for i in range(0, len(flattened_captions), 2):
        random_num = random.random()
        if random_num < 0.25:
            captions.append(
                f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}")
        elif 0.25 < random_num < 0.5:
            captions.append(
                f"{flattened_captions[i + 1].strip('.?, ').capitalize()} and {flattened_captions[i].strip('.?, ')}")
        elif 0.5 < random_num < 0.75:
            captions.append(f"{flattened_captions[i].strip('.?, ').capitalize()}")
        else:
            captions.append(f"{flattened_captions[i + 1].strip('.?, ').capitalize()}")
    return captions


def fiq_collate_fn_train(batch):
    ref_name, ref_img, captions, target_name = default_collate(batch)
    # process captions
    flattened_captions = np.array(captions).T.flatten().tolist()
    combined_captions = combine_captions(flattened_captions)
    return ref_name, ref_img, combined_captions, target_name


def fiq_collate_fn_val(batch):
    candidate_name, candidate_image, captions, target_name = default_collate(batch)
    # process captions
    flattened_captions = np.array(captions).T.flatten().tolist()
    combined_captions = [
        f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
        i in range(0, len(flattened_captions), 2)]
    return candidate_name, candidate_image, combined_captions, target_name


def build_data(name: str, bs, preprocess):
    try:
        assert name in ['fiq-dress', 'fiq-shirt', 'fiq-toptee', 'cirr']
    except AssertionError:
        print(f"Unknown dataset name {name}. Dataset name must be one of 'fiq-dress', 'fiq-shirt', 'fiq-toptee', 'cirr'.")

    if name.startswith('fiq'):
        source_data = name.split('-')[-1]
        datasets = [
            FashionIQDataset(
                mode='query', 
                clothtype=source_data, 
                split='train', 
                path='./fashion-iq',
                preprocess=preprocess
            ),
            FashionIQDataset(
                mode='query', 
                clothtype=source_data, 
                split='val', 
                path='./fashion-iq',
                preprocess=preprocess
            ),
            FashionIQDataset(
                mode='target', 
                clothtype=source_data,
                path='./fashion-iq',
                preprocess=preprocess
            )
        ]

    elif name == 'cirr':
        datasets = [
            CIRRDataset(
                mode='query', 
                split='train', 
                path='./cirr',
                preprocess=preprocess
            ),
            CIRRDataset(
                mode='query', 
                split='val', 
                path='./cirr',
                preprocess=preprocess
            ),
            CIRRDataset(
                mode='target', 
                path='./cirr',
                preprocess=preprocess
            )
        ]

    train_ds, val_ds, tgt_ds = datasets
    return [
        DataLoader(
            dataset=train_ds,
            batch_size=bs,
            num_workers=16,
            pin_memory=False,
            drop_last=True,
            shuffle=True,
            collate_fn = fiq_collate_fn_train if name.startswith('fiq') else default_collate
        ),
        DataLoader(
            dataset=val_ds,
            batch_size=bs,
            num_workers=16,
            pin_memory=False,
            drop_last=True,
            shuffle=False,
            collate_fn = fiq_collate_fn_val if name.startswith('fiq') else default_collate
        ),
        DataLoader(
            dataset=tgt_ds,
            batch_size=128
        )
    ]