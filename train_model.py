#!/nfs/polizzi/bfry/programs/miniconda3/envs/ligandmpnn_env/bin/python

import random
import numpy as np
from typing import *

import torch
from model_utils import ProteinMPNN
from torch.utils.data import DataLoader

from training.database import UnclusteredProteinChainDataset, LigandMPNNDatasetSampler, collate_cluster_samples


database_params = {
    'debug': True,
    'raw_dataset_path': '/nfs/polizzi/bfry/laser_training_database/all_data_shelf_hbond_sconly_rigorous.db',
    'metadata_dataset_path': '/nfs/polizzi/bfry/laser_training_database/pdb_metadata_shelf_addhaslig_perchain.db',
    'clustering_dataframe_path': '/nfs/polizzi/bfry/laser_training_database/sequence_split_clusters_with_structural_contam.pkl',

    'batch_size': 6_000,
    'sample_randomly': True,
    'max_protein_length': 6_000,
    'use_xtal_additive_ligands': True,
}


model_hyperparams = {
    'node_features': 128, 
    'edge_features': 128, 
    'hidden_dim': 128, 
    'num_encoder_layers': 3, 
    'num_decoder_layers': 3, 
    'k_neighbors': 32, 
    'vocab': 21, 
    'ligand_mpnn_use_side_chain_context': True,
    'cutoff_for_score': 8.0,
    'use_atom_context': True,
    'atom_context_num': 24, 
}

def set_random_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dataloaders(database: UnclusteredProteinChainDataset, seed: int) -> Tuple[DataLoader, DataLoader]:
    collate_fn = lambda x: collate_cluster_samples(x, database_params['use_xtal_additive_ligands'], model_hyperparams['cutoff_for_score'], model_hyperparams['use_atom_context'], model_hyperparams['atom_context_num'])

    train_sampler = LigandMPNNDatasetSampler(database, database_params, is_train=True, seed=0, max_protein_length=database_params['max_protein_length'])
    train_dataloader = DataLoader(database, collate_fn=collate_fn, batch_sampler=train_sampler)

    test_sampler = LigandMPNNDatasetSampler(database, database_params, is_train=False, seed=0, max_protein_length=database_params['max_protein_length'])
    test_dataloader = DataLoader(database, collate_fn=collate_fn, batch_sampler=test_sampler)

    return train_dataloader, test_dataloader



def send_tensors_to_device(batch_dict: dict, device: torch.device) -> dict:
    output = {}
    for i,j in batch_dict.items():
        if isinstance(j, torch.Tensor):
            output[i] = j.to(device)
        else:
            output[i] = j
    return output


@torch.no_grad()
def sample_model(model: ProteinMPNN, batch_dict: dict):
    model.eval()

    batch_dict['randn'] = torch.randn(batch_dict['mask_XY'].shape, device=model.device)
    batch_dict['temperature'] = 1.0

    batch_dict = send_tensors_to_device(batch_dict, model.device)
    sample_output = model.sample(batch_dict)

    return sample_output


def train_model(model: ProteinMPNN, batch_dict: dict):
    model.train()

    batch_dict = send_tensors_to_device(batch_dict, model.device)
    batch_dict['randn'] = torch.randn(batch_dict['mask_XY'].shape, device=model.device)

    logits = model(batch_dict)

    raise NotImplementedError


def main(device_str: str, seed=0):
    set_random_seeds(seed)
    device = torch.device(device_str)

    model = ProteinMPNN(**model_hyperparams).to(device)
    database = UnclusteredProteinChainDataset(database_params)
    train_dataloader, test_dataloader = get_dataloaders(database, seed=seed)

    for batch in train_dataloader:
        train_model(model, batch)
        break



if __name__ == "__main__":
    main('cuda:4')