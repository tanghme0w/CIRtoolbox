from .data import (
    FashionIQDataset, 
    CIRRDataset,
    fiq_collate_fn_train, 
    fiq_collate_fn_val
)
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader, default_collate


def build_ddp_data(name: str, bs, rank, world_size, preprocess):
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

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    )
    
    return [
        DataLoader(
            sampler=train_sampler,
            dataset=train_ds,
            batch_size=bs,
            num_workers=16,
            pin_memory=False,
            drop_last=True,
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
