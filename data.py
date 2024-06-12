from torch.utils.data import Dataset
import torch
import os
import json
from PIL import Image


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
            image = self.preprocess(Image.open(image_file))
            return image_name, image
        
        elif self.mode == 'query':
            candidate_name = self.metadata[index]['candidate']
            target_name = self.metadata[index]['target']
            captions = self.metadata[index]['captions']
            candidate_image_file = os.path.join(self.image_path, f'{candidate_name}.png')
            candidate_image = self.preprocess(Image.open(candidate_image_file))
            
            if self.split == 'train':
                target_image_file = os.path.join(self.image_path, f'{target_name}.png')
                target_image = self.preprocess(Image.open(target_image_file))
                return candidate_name, candidate_image, captions, target_image
            else:
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
        self.triplets = json.load(open(os.path.join(path, f'cirr/captions/cap.rc2.{self.split}.json')))
        self.namepath = json.load(open(os.path.join(path, f'cirr/image_splits/split.rc2.{split}.json')))
    

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