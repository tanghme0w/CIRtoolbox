import torch
from model import *
from data import build_data
from tqdm import tqdm
import orjson
import os


def showcase(checkpoint, dataset, bs, outfile, device):
    """ load model """
    model = CLIPSumModelEVA().to(device)
    ckpt = torch.load(checkpoint)
    model.load_state_dict(ckpt['model_sd'])

    """ load dataset """
    try:
        _, val_loader, target_loader = build_data(dataset, bs, model.preprocess)
    except AttributeError:
        _, val_loader, target_loader = build_data(dataset, bs)

    """ infer """
    model.eval()

    """ initialize metric and case dict """
    metrics = [10, 50]
    recall = dict()
    good_case = dict()
    bad_case  = dict()
    for metric in metrics:
        recall[metric] = 0
        good_case[metric] = []
        bad_case[metric] = []
    sample_count = 0

    with torch.no_grad():

        """ get all target feature from validation set """
        target_bar = tqdm(target_loader, desc="[Validate] Target Feature")
        all_target_names = []
        all_target_features = []
        for img_name, img in target_bar:
            all_target_names += img_name
            all_target_features.append(model.target_forward(img))
        all_target_features = torch.cat(all_target_features)

        """ get all query feature for validation """
        val_bar = tqdm(val_loader, desc="[Validate] Query Feature")
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
                # sort out good case and bad case
                for i, label in enumerate(labels):
                    if torch.sum(label[ :metric]) >= 1:
                        good_case[metric].append(ref_name[i])
                    else:
                        bad_case[metric].append(ref_name[i])
            sample_count += len(ref_img)
        for i in metrics:
            recall[i] /= sample_count
            recall[i] *= 100
        print([f"R@{i}: {recall[i]}" for i in metrics])

    """ write outfile """
    with open(outfile, "a") as of:
        for metric in metrics:
            result = orjson.dumps({
                'checkpoint': checkpoint,
                'metric': metric,
                'recall': float(recall[metric]),
                'good_case': good_case[metric],
                'bad_case': bad_case[metric],
            }).decode('utf-8')
            of.write(result + '\n')

def _get_all_file_paths(root_dir):
    file_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # Construct the absolute file path
            absolute_path = os.path.join(dirpath, filename)
            file_paths.append(absolute_path)
    return file_paths
    
def showcase_all_data(root_dir, ds, outfile, bs=256, device='cuda'):
    """ Showcase all checkpoint for a specific dataset

    Args:
        root_dir (str): directory containing all checkpoint
        ds (str): dataset name
    """
    checkpoints = [filepath for filepath in _get_all_file_paths(root_dir) \
                   if ds in filepath and filepath.endswith('.pt')]
    
    for checkpoint in checkpoints:
        showcase(
            checkpoint=checkpoint, 
            dataset=ds, 
            bs=bs, 
            device=device, 
            outfile=outfile
        )


if __name__ == '__main__':
    """ Test showcase """
    # showcase(
    #     checkpoint='saved/20240814-23-16-20_fiq-dress/epoch001.pt',
    #     dataset='fiq-dress',
    #     bs=256,
    #     outfile='case_study/dress.jsonl',
    #     device='cuda'
    # )

    """ Test showcase_all_data """
    showcase_all_data(
        root_dir='saved',
        ds='fiq-dress',
        outfile='case_study/fiq.jsonl'
    )