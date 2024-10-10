#!/nfs/polizzi/bfry/programs/miniconda3/envs/ligandmpnn_env/bin/python

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

def send_tensors_to_device(batch_dict: dict, device: torch.device) -> dict:
    output = {}
    for i,j in batch_dict.items():
        if isinstance(j, torch.Tensor):
            output[i] = j.to(device)
        else:
            output[i] = j
    return output

def main():
    device = torch.device('cpu')
    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        vocab=21,
        device=device,
        atom_context_num=24,
        model_type='ligand_mpnn',
        ligand_mpnn_use_side_chain_context=True,
    ).to(device)
    model.train()

    training_database = UnclusteredProteinChainDataset(database_params)
    train_sampler = LigandMPNNDatasetSampler(training_database, database_params, is_train=True, seed=0, max_protein_length=database_params['max_protein_length'])

    collate_fn = lambda x: collate_cluster_samples(x, database_params['use_xtal_additive_ligands'])
    train_dataloader = DataLoader(training_database, collate_fn=collate_fn, batch_sampler=train_sampler)

    for batch in train_dataloader:

        batch['randn'] = torch.randn(batch['mask_XY'].shape, device=model.device)
        batch['temperature'] = 1.0
        batch = send_tensors_to_device(batch, model.device)
        sample_output = model(batch)
        break


    data = [training_database[0], training_database[1]]



if __name__ == "__main__":

    main()