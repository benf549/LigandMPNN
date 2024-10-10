import os
import shelve
import ast
import pandas as pd
from typing import *
from collections import defaultdict
from dataclasses import dataclass

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from data_utils import element_list, batch_get_nearest_neighbours

MAX_PEPTIDE_LENGTH = 40
MIN_TM_SCORE_FOR_SIMILARITY = 0.70

aa_short_to_long = {'C': 'CYS', 'D': 'ASP', 'S': 'SER', 'Q': 'GLN', 'K': 'LYS', 'I': 'ILE', 'P': 'PRO', 'T': 'THR', 'F': 'PHE', 'N': 'ASN', 'G': 'GLY', 'H': 'HIS', 'L': 'LEU', 'R': 'ARG', 'W': 'TRP', 'A': 'ALA', 'V': 'VAL', 'E': 'GLU', 'Y': 'TYR', 'M': 'MET', 'X': 'XAA'}
aa_long_to_short = {x: y for y, x in aa_short_to_long.items()}
aa_long_to_idx = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLU': 5, 'GLN': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19, 'XAA': 20}
aa_short_to_idx = {x: aa_long_to_idx[y] for x, y in aa_short_to_long.items()}
aa_idx_to_short = {v: k for k, v in aa_short_to_idx.items()}

restype_STRtoINT = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
    "X": 20,
}

dataset_atom_order = {
    'G': ['N', 'CA', 'C', 'O'],
    'X': ['N', 'CA', 'C', 'O'],
    'A': ['N', 'CA', 'C', 'O', 'CB'],
    'S': ['N', 'CA', 'C', 'O', 'CB', 'OG', 'HG'],
    'C': ['N', 'CA', 'C', 'O', 'CB', 'SG', 'HG'],
    'T': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', 'HG1'],
    'P': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
    'V': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],
    'M': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
    'N': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
    'I': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
    'L': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
    'D': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
    'E': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
    'K': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
    'Q': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
    'H': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', 'HD1', 'HE2'],
    'F': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'R': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
    'Y': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', 'HH'],
    'W': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'NE1', 'CZ2', 'CZ3', 'CH2']
}

aa_to_chi_angle_atom_map = {
    'C': {1: ('N', 'CA', 'CB', 'SG'), 2: ('CA', 'CB', 'SG', 'HG')},
    'D': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'OD1')},
    'E': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD'), 3: ('CB', 'CG', 'CD', 'OE1')},
    'F': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD1')},
    'H': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'ND1')},
    'I': {1: ('N', 'CA', 'CB', 'CG1'), 2: ('CA', 'CB', 'CG1', 'CD1')},
    'K': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD'), 3: ('CB', 'CG', 'CD', 'CE'), 4: ('CG', 'CD', 'CE', 'NZ')},
    'L': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD1')},
    'M': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'SD'), 3: ('CB', 'CG', 'SD', 'CE')},
    'N': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'OD1')},
    'P': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD')},
    'Q': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD'), 3: ('CB', 'CG', 'CD', 'OE1')},
    'R': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD'), 3: ('CB', 'CG', 'CD', 'NE'), 4: ('CG', 'CD', 'NE', 'CZ')},
    'S': {1: ('N', 'CA', 'CB', 'OG'), 2: ('CA', 'CB', 'OG', 'HG')},
    'T': {1: ('N', 'CA', 'CB', 'OG1'), 2: ('CA', 'CB', 'OG1', 'HG1')},
    'V': {1: ('N', 'CA', 'CB', 'CG1')},
    'W': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD1')},
    # NOTE: need to align leftover atoms to the first two chi angles before the final chi angle only for TYR.
    'Y': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD1'), 3: ('CE1', 'CZ', 'OH', 'HH')}
}

element_dict = dict(zip(element_list, range(1, len(element_list))))

##### Converts the atom names defined above to indices in the dataset atom order tensor
#### Assumes tensors are padded with with NaN at MAX_NUM_RESIDUE_ATOMS'th index in dim 1 
MAX_NUM_RESIDUE_ATOMS = max([len(res) for res in dataset_atom_order.values()])
placeholder_indices = torch.tensor([MAX_NUM_RESIDUE_ATOMS] * 4)
aa_to_chi_angle_atom_index = torch.full((20, 4, 4), MAX_NUM_RESIDUE_ATOMS)
aa_to_chi_angle_mask = torch.full((21, 4), False)
aa_to_leftover_atoms = torch.full((20, MAX_NUM_RESIDUE_ATOMS), MAX_NUM_RESIDUE_ATOMS)
# Iterate in the order of the canonical amino acid indices in aa_idx_to_short
for idx in range(21):
    if idx == 20:
        aa = 'G'
    else:
        aa = aa_idx_to_short[idx]
    if aa in aa_to_chi_angle_atom_map:
        # Fill residues that have chi angles with indices of relevant atoms.
        all_atoms_set = set([x for x in range(len(dataset_atom_order[aa]))])
        chi_placed_atoms_set = set()
        for chi_num, atom_names in aa_to_chi_angle_atom_map[aa].items():
            chi_placed_indices = [dataset_atom_order[aa].index(x) for x in atom_names]
            aa_to_chi_angle_atom_index[idx, chi_num - 1] = torch.tensor(chi_placed_indices)
            chi_placed_atoms_set.update(chi_placed_indices)

        # Track which atoms are not involved in the placement process
        leftovers = sorted(list(all_atoms_set - chi_placed_atoms_set - {2, 3}))
        aa_to_leftover_atoms[idx, :len(leftovers)] = torch.tensor(leftovers)
        
        # Fill mask with True for chi angles and False for padding.
        aa_to_chi_angle_mask[idx, :len(aa_to_chi_angle_atom_map[aa])] = True


atomic_number_to_atom = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt', 110: 'Ds', 111: 'Rg', 112: 'Cn', 113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'}

curr_path = os.path.dirname(os.path.abspath(__file__))
ligandmpnn_training_pdb_codes = ast.literal_eval(open(os.path.join(curr_path, f'train.json'), 'r').read().strip())
ligandmpnn_validation_pdb_codes = ast.literal_eval(open(os.path.join(curr_path, f'valid.json'), 'r').read().strip())
ligandmpnn_test_sm_pdb_codes = ast.literal_eval(open(os.path.join(curr_path, f'test_small_molecule.json'), 'r').read().strip())
ligandmpnn_test_nucleotide = ast.literal_eval(open(os.path.join(curr_path, f'test_nucleotide.json'), 'r').read().strip())
ligandmpnn_test_metal = ast.literal_eval(open(os.path.join(curr_path, f'test_metal.json'), 'r').read().strip())

def chain_list_to_protein_chain_dict(chain_list: list) -> dict:
    """
    Takes a list of bioassemblies+segment+chains and returns a dictionary 
    mapping pdb code to a list of assemblies and chains in a given sequence cluster.
    """

    bioasmb_list = defaultdict(list)
    for chain in chain_list:
        pdb_code, asmb_chain_id = chain.split('_')
        bioasmb_list[pdb_code].append(asmb_chain_id)

    return dict(bioasmb_list)


def invert_dict(d: dict) -> dict:
    clusters = defaultdict(list)
    for k, v in d.items():
        clusters[v].append(k)
    return dict(clusters)


def pdb_and_chain_to_code(pdb_code: str, chain_tup: Tuple[str, str]) -> str:
    return '-'.join([pdb_code, *chain_tup])


def get_complex_len(complex_data: dict) -> int:
    """
    Chains are indicated with segment/chain tuples so only these have size attr.
    """
    
    # Create a sorta-resnum by adding every 10 ligand atoms to the resnum count.
    num_lig_coords = 0
    if 'ligands' in complex_data:
        num_lig_coords += sum(len(x) for x in complex_data['ligands']['coords'])
    if 'xtal_additives' in complex_data:
        num_lig_coords += sum(len(x) for x in complex_data['xtal_additives']['coords'])

    num_protein_nodes = sum([x['size'] for y, x in complex_data.items() if isinstance(y, tuple)])

    return num_protein_nodes + num_lig_coords


def compute_metadata_from_raw_data_shelve(path_to_raw_shelve: str, path_to_output_shelve: str, is_debug: bool) -> None:
    """
    Computes metadata from the raw data shelve and creates a new shelve storing that metadata.
    """
    chain_key_to_index = {}
    index_to_complex_size = {}
    index_to_num_ligand_contacting_residues = {}

    idx = 0
    with shelve.open(path_to_raw_shelve, 'r', protocol=5) as db_:
        # Only load the debug chains if we are in debug mode.
        db_keys = list(db_.keys())
        if is_debug:
            db_keys = [x for x in db_keys if x[1:3] == 'w7' or x[:4] == '4jnj']

        # Loop over all the chains and record the chain key and the complex size.
        for pdb_code in tqdm(db_keys, desc='Computing metadata for dataset sampling...', dynamic_ncols=True):
            protein_data = db_[pdb_code]
            protein_complex_len = get_complex_len(protein_data)

            for chain_key, chain_data in protein_data.items():
                if isinstance(chain_key, tuple):
                    chain_key = '-'.join([pdb_code] + list(chain_key))
                    chain_key_to_index[chain_key] = idx
                    index_to_complex_size[idx] = protein_complex_len
                    index_to_num_ligand_contacting_residues[idx] = chain_data['first_shell_ligand_contact_mask'].sum()
                    idx += 1

    # Invert the chain_key_to_index dict to get a cluster to chain mapping.
    index_to_chain_key = {x: y for y,x in chain_key_to_index.items()}

    # Write the metadata to a shelve.
    with shelve.open(path_to_output_shelve, 'c', protocol=5) as db_:
        db_['chain_key_to_index'] = chain_key_to_index
        db_['index_to_complex_size'] = index_to_complex_size
        db_['index_to_chain_key'] = index_to_chain_key
        db_['index_to_num_ligand_contacting_residues'] = index_to_num_ligand_contacting_residues


class UnclusteredProteinChainDataset(Dataset):
    """
    Dataset where every pdb_assembly-segment-chain is a separate index.
    """
    def __init__(self, params):

        metadata_shelve_path = params['metadata_dataset_path'] + ('.debug' if params['debug'] else '')
        if not os.path.exists(metadata_shelve_path + '.dat'):
            print("Computing dataset metadata shelve, this only needs to run once.")
            compute_metadata_from_raw_data_shelve(params['raw_dataset_path'], metadata_shelve_path, params['debug'])

        self.pdb_code_to_complex_data = shelve.open(params['raw_dataset_path'], 'r', protocol=5)

        metadata = shelve.open(metadata_shelve_path, 'r', protocol=5)
        self.chain_key_to_index = metadata['chain_key_to_index']
        self.index_to_complex_size = metadata['index_to_complex_size']
        self.index_to_chain_key = metadata['index_to_chain_key']
        self.index_to_num_ligand_contacting_residues = metadata['index_to_num_ligand_contacting_residues']
        metadata.close()

    def __del__(self) -> None:
        if hasattr(self, 'pdb_code_to_complex_data'):
            self.pdb_code_to_complex_data.close()

    def __len__(self) -> int:
        return len(self.chain_key_to_index)

    def __getitem__(self, index: int) -> Tuple[dict, str]:
        # Take indexes unique to chain and return the complex data for that chain and the chain key.
        chain_key = self.index_to_chain_key[index]
        pdb_code = chain_key.split('-')[0]
        output_data = self.pdb_code_to_complex_data[pdb_code]
        return output_data, chain_key
    
    def write_all_sequence_fasta(self, output_path: str) -> None:
        """
        Writes all sequences longer than MAX_PEPTIDE_LENGTH to a fasta file.

        Run 30% cluster generation with:
            `mmseqs easy-cluster fasta.txt cluster30test tmp30test --min-seq-id 0.3 -c 0.5 --cov-mode 5 --cluster-mode 3`
        """
        output = {}
        # Loop over everything in the dataset.
        for pdb_code, data_dict in tqdm(self.pdb_code_to_complex_data.items(), total=len(self.pdb_code_to_complex_data)):
            for key, sub_data in data_dict.items():
                # Select the chains which are ('Segment', 'Chain') tuples and record crystallized sequence.
                if isinstance(key, tuple):
                    chain_key = "-".join([pdb_code, *key])
                    sequence = sub_data['polymer_seq']
                    output[chain_key] = sequence
        
        # Sort the output by chain_key so the fasta file is sorted.
        output = sorted(output.items(), key=lambda x: x[0])

        # Write the fasta file.
        with open(output_path, 'w') as f:
            for chain_key, sequence in output:
                if sequence is not None and len(sequence) > MAX_PEPTIDE_LENGTH:
                    f.write(f">{chain_key}\n")
                    f.write(f"{sequence}\n")


class LigandMPNNDatasetSampler(Sampler):
    """
    Samples a single protein complex from precomputed mmseqs clusters.
    Ensures samples drawn evenly by sampling first from sequence clusters, then by pdb_code, then by assembly and chain.
    Iteration returns batched indices for use in UnclusteredProteinChainDataset.
    Pass to a DataLoader as a batch_sampler.
    """
    def __init__(self, dataset: UnclusteredProteinChainDataset, params: dict, is_train: bool, seed: Optional[int] = None, max_protein_length: int = 10_000):
        # Set the random seed for reproducibility and consistent randomness between processes if parallelized.
        if seed is None:
            self.generator = torch.Generator(device='cpu')
        else:
            self.generator = torch.Generator().manual_seed(seed)

        # The unclustered dataset where each complex/assembly is a single index.
        self.dataset = dataset
        self.batch_size = params['batch_size']
        self.shuffle = params['sample_randomly']
        self.max_protein_length = max_protein_length

        # Load the cluster data.
        sequence_clusters = pd.read_pickle(params['clustering_dataframe_path'])

        # if params['debug']:
        #     sequence_clusters = sequence_clusters[sequence_clusters.chain.str.find('w7') == 1]
        
        self.is_train = is_train
        self.train_codes = ligandmpnn_training_pdb_codes
        self.val_codes = ligandmpnn_validation_pdb_codes
        self.test_codes = ligandmpnn_test_sm_pdb_codes

        # Maps sequence cluster to number of chains and vice versa
        self.chain_to_cluster = sequence_clusters.set_index('chain').to_dict()['cluster_representative']
        self.cluster_to_chains = invert_dict(self.chain_to_cluster)
        
        # Load relevant pickled sets of cluster keys, filter for train/test as necessary.
        self.cluster_to_chains = self.filter_clusters()

        # Sample for the first epoch, subsequent epochs will resample after iteration over samples is complete.
        self.curr_samples = []
        self.sample_clusters()

        self.curr_batches = []
        self.construct_batches()

    def __len__(self) -> int:
        """
        Returns number of batches in the current epoch.
        """
        return len(self.curr_batches)

    def filter_clusters(self) -> dict:
        """
        Filter clusters based on the given dataset sampler and the max protein length.
            Parameters:
            - is_test_dataset_sampler (bool): True if the dataset sampler is for the test dataset, False otherwise.

            Returns:
            - dict: A dictionary containing the filtered clusters.
        """

        val_code_set = set(self.val_codes)
        val_test_pdb_codes = set(ligandmpnn_validation_pdb_codes + ligandmpnn_test_sm_pdb_codes + ligandmpnn_test_metal + ligandmpnn_test_nucleotide)

        # Get the cluster names containing any val or test pdb code information.
        contaminated_clusters = set()
        for cluster, pdb_code_list in self.cluster_to_chains.items():
            if any([x.split('_')[0] in val_test_pdb_codes for x in pdb_code_list]):
                contaminated_clusters.add(cluster)

        non_contam_clusters = {k: v for k,v in self.cluster_to_chains.items() if k not in contaminated_clusters}

        output = defaultdict(list)
        if self.is_train:
            for cluster_rep, cluster_list in non_contam_clusters.items():
                for chain in cluster_list:
                    if chain not in self.dataset.chain_key_to_index:
                        continue
                    chain_len = self.dataset.index_to_complex_size[self.dataset.chain_key_to_index[chain]]
                    if chain_len <= self.max_protein_length:
                        output[cluster_rep].append(chain)
        else:
            potential_test_clusters = {k: v for k,v in self.cluster_to_chains.items() if k in contaminated_clusters}
            for cluster_rep, cluster_list in potential_test_clusters.items():
                for chain in cluster_list:
                    if chain not in self.dataset.chain_key_to_index or chain.split('_')[0] not in val_code_set:
                        continue
                    chain_len = self.dataset.index_to_complex_size[self.dataset.chain_key_to_index[chain]]
                    if chain_len <= self.max_protein_length:
                        output[cluster_rep].append(chain)
        return output
        
    
    def sample_clusters(self) -> None:
        """
        Randomly samples clusters from the dataset for the next epoch.
        Updates the self.curr_samples list with new samples.
        """
        self.curr_samples = []
        # Loop over mmseqs cluster and list of chains for that cluster.
        for cluster, chains in self.cluster_to_chains.items():
            # Convert list of all chains/pdbs/assemblies to a dictionary mapping pdb code to a 
            # list of assemblies and chains in the current cluster.
            pdb_to_assembly_chains_map = chain_list_to_protein_chain_dict(chains)

            # Sample from the PDBs with the desired chain cluster.
            sample_index = int(torch.randint(0, len(pdb_to_assembly_chains_map), (1,), generator=self.generator).item())
            sampled_pdb = list(pdb_to_assembly_chains_map.keys())[sample_index]

            # Given the PDB to sample from sample an assembly and chain for training.
            asmbly_index = int(torch.randint(0, len(pdb_to_assembly_chains_map[sampled_pdb]), (1,), generator=self.generator).item())
            sampled_assembly_and_chains = pdb_to_assembly_chains_map[sampled_pdb][asmbly_index]
    
            # Reform the string representation of the sampled pdb_assembly-seg-chain.
            chain_key = '_'.join([sampled_pdb, sampled_assembly_and_chains])

            # Yield the index of the sampled pdb_assembly-seg-chain.
            self.curr_samples.append(self.dataset.chain_key_to_index[chain_key])

    def construct_batches(self):
        """
        Batches by size inspired by:
            https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler:~:text=%3E%3E%3E%20class%20AccedingSequenceLengthBatchSampler
        """
        # Reset the current batches.
        self.curr_batches = []

        # Sort the samples by size.
        curr_samples_tensor = torch.tensor(self.curr_samples)
        sizes = torch.tensor([self.dataset.index_to_complex_size[x] for x in self.curr_samples])
        size_sort_indices = torch.argsort(sizes)

        # iterate through the samples in order of size, create batches of size batch_size.
        debug_sizes = []
        curr_list_sample_indices, curr_list_sizes = [], []
        for curr_size_sort_index in size_sort_indices:
            # Get current sample index and size.
            curr_sample_index = curr_samples_tensor[curr_size_sort_index].item()
            curr_size = sizes[curr_size_sort_index].item()

            # Add to the current batch if would not exceed batch size otherwise create a new batch.
            if sum(curr_list_sizes) + curr_size <= self.batch_size:
                curr_list_sample_indices.append(curr_sample_index)
                curr_list_sizes.append(curr_size)
            else:
                # Add the current batch to the list of batches.
                self.curr_batches.append(curr_list_sample_indices)
                debug_sizes.append(sum(curr_list_sizes))

                # Reset the current batch.
                curr_list_sizes = [curr_size]
                curr_list_sample_indices = [curr_sample_index]

        # Store any remaining samples.
        if len(curr_list_sample_indices) > 0:
            self.curr_batches.append(curr_list_sample_indices)
            debug_sizes.append(sum(curr_list_sizes))

        # Shuffle the batches.
        if self.shuffle:
            shuffle_indices = torch.randperm(len(self.curr_batches), generator=self.generator).tolist()
            curr_batches_ = [self.curr_batches[x] for x in shuffle_indices]
            self.curr_batches = curr_batches_

        # Sanity check that we have the correct number of samples after iteration.
        assert sum(debug_sizes) == sizes.sum().item(), "Mismatch between number of samples and expected size of samples."

    def __iter__(self):
        # Yield the batches we created.
        for batch in self.curr_batches:
            yield batch

        # Resample for the next epoch, and create new batches.
        self.sample_clusters()
        self.construct_batches()


def collate_cluster_samples(data: list, use_xtal_additive_ligands: bool, cutoff_for_score: float, use_atom_context: bool, atom_context_num: int):
    """
    Given a list of data from protein dataset, creates a BatchData object holding a single batch for input to model.
    Handles converting protein complex data into torch tensors with desired shapes and properties.
    """
    L_max = 0
    Y_max = 0
    chain_idx = 0
    all_batch_data = defaultdict(list)
    for batch_idx, (complex_data, chain_key) in enumerate(data):


        try:
            all_lig_coords = []
            all_lig_elements = []
            if 'ligands' in complex_data:
                for (lig_coords, lig_elements) in zip(complex_data['ligands']['coords'], complex_data['ligands']['elements']):

                    if 'X' in lig_elements:
                        continue

                    all_lig_coords.append(lig_coords)
                    all_lig_elements.extend([element_dict[''.join([y.upper() for y in x])] for x in lig_elements])
            
            if 'xtal_additives' in complex_data and use_xtal_additive_ligands:
                for (lig_coords, lig_elements) in zip(complex_data['xtal_additives']['coords'], complex_data['xtal_additives']['elements']):
                    if 'X' in lig_elements:
                        continue

                    all_lig_coords.append(lig_coords)
                    all_lig_elements.extend([element_dict[''.join([y.upper() for y in x])] for x in lig_elements])
            
            # Construct the ligand data.
            Y = torch.cat(all_lig_coords, dim=0)
            Y_t = torch.tensor(all_lig_elements, dtype=torch.long)
            Y_m = (Y_t != 1) * (Y_t != 0)

            Y = Y[Y_m, :]
            Y_t = Y_t[Y_m]
            Y_m = Y_m[Y_m]

        except:
            print('Error in ligand data, defaulting to empty tensors.')
            Y = torch.zeros((1, 3), dtype=torch.float)
            Y_t  = torch.zeros((1,), dtype=torch.long)
            Y_m = torch.zeros((1,), dtype=torch.long)
        
        curr_y = Y.shape[0]
        Y_max = max(Y_max, curr_y)
        
        # Single-chain complexes don't have TM-align data.
        tm_align_map = None
        if chain_key in complex_data['tm_align_scores']:
            tm_align_map = complex_data['tm_align_scores'][chain_key]
        
        curr_pdb_code = complex_data['pdb_code']

        curr_l = 0
        r_idx_offset = 0
        all_complex_data = defaultdict(list)
        for curr_chain_tup, chain_data in complex_data.items():

            # Chains are represented with key tuples of (segment, chain), assume everything else is metadata.
            if not isinstance(curr_chain_tup, tuple):
                continue

            # Convert chain tuple to chain key by appending pdb code and concatenating with '-'.
            curr_chain_key = pdb_and_chain_to_code(curr_pdb_code, curr_chain_tup)

            # If missing TM-align data, set to 1.0 so we mask it by default.
            sampled_chain_to_curr_chain_tm_align_score = 1.0
            if (not tm_align_map is None) and (curr_chain_key in tm_align_map):
                sampled_chain_to_curr_chain_tm_align_score = tm_align_map[curr_chain_key]

            # Provides rotamers for anything not structurally similar, or smaller in size than MAX_PEPTIDE_LENGTH.
            # Recently, anything smaller than MAX_PEPTIDE_LENGTH should have been converted into a ligand.
            if sampled_chain_to_curr_chain_tm_align_score < MIN_TM_SCORE_FOR_SIMILARITY or chain_data['size'] <= MAX_PEPTIDE_LENGTH:
                chain_mask = torch.ones(chain_data['size'], dtype=torch.bool)
            else:
                chain_mask = torch.zeros(chain_data['size'], dtype=torch.bool)
            # Flip the chain mask and make numeric since proteinmpnn uses false for fixed residues and True for variable.
            chain_mask = (~chain_mask).long()

            # Precomputed in ProteinAssemblyDataset.compute_masks
            first_shell_ligand_contact_mask = chain_data['first_shell_ligand_contact_mask']
            extra_atom_contact_mask = chain_data['extra_atom_contact_mask']

            # Extract the rest of the chain data.
            sequence_indices = chain_data['sequence_indices'].long()
            resnum_indices = chain_data['seqres_resnums']
            chi_angles = chain_data['chi_angles']

            # Compute a mask for chi_angles that are missing, second chi angle for Cys to be absent.
            non_nan_chi_mask = (~chi_angles.isnan())
            cys_mask = (sequence_indices == aa_short_to_idx['C'])
            expected_num_chi_mask = (non_nan_chi_mask.sum(dim=-1) == aa_to_chi_angle_mask[sequence_indices].sum(dim=-1))
            expected_num_chi_mask[cys_mask] = non_nan_chi_mask[cys_mask, 0]

            # Set malformed residues to X, Remove the chi angles for residues that are missing them.
            sequence_indices[~expected_num_chi_mask] = aa_short_to_idx['X']
            chi_angles[~expected_num_chi_mask, :] = torch.nan

            # Our dataset is stored as (N, CA, CB, C, O), ligandmpnn expects (N, Ca, C, CB, O)
            backbone_coords = chain_data['backbone_coords'].float()
            CB_coords = backbone_coords[:, 2]
            backbone_coords = backbone_coords[:, [0, 1, 3, 2, 4]]

            if resnum_indices is None or resnum_indices.numel() == 0:
                resnum_indices = torch.arange((backbone_coords.shape[0],), dtype=torch.long)

            xyz_37 = F.pad(backbone_coords, (0, 0, 0, 37 - backbone_coords.shape[1], 0, 0), 'constant', value=torch.nan)
            xyz_37_m = (~(xyz_37.isnan())).any(dim=-1).long()
            mask = (~torch.isnan(backbone_coords).any(dim=-1).any(dim=-1)).long()
            chain_labels = torch.full((backbone_coords.shape[0],), chain_idx, dtype=torch.long)
            S = sequence_indices
            R_idx = resnum_indices + r_idx_offset
            r_idx_offset += len(resnum_indices)

            all_complex_data['X'].append(backbone_coords[:, [0, 1, 2, 4]])
            all_complex_data['xyz_37'].append(xyz_37)
            all_complex_data['xyz_37_m'].append(xyz_37_m)
            all_complex_data['mask'].append(mask) 
            all_complex_data['chain_labels'].append(chain_labels)
            all_complex_data['chain_mask'].append(chain_mask)
            all_complex_data['S'].append(S)
            all_complex_data['Y'].append(Y)
            all_complex_data['Y_t'].append(Y_t)
            all_complex_data['Y_m'].append(Y_m)
            all_complex_data['R_idx'].append(R_idx)

            chain_idx += 1
            curr_l += backbone_coords.shape[0]

        all_batch_data['X'].append(torch.cat(all_complex_data['X'], dim=0))
        all_batch_data['xyz_37'].append(torch.cat(all_complex_data['xyz_37'], dim=0))
        all_batch_data['xyz_37_m'].append(torch.cat(all_complex_data['xyz_37_m'], dim=0))
        all_batch_data['mask'].append(torch.cat(all_complex_data['mask'], dim=0))
        all_batch_data['chain_labels'].append(torch.cat(all_complex_data['chain_labels'], dim=0))
        all_batch_data['chain_mask'].append(torch.cat(all_complex_data['chain_mask'], dim=0))
        all_batch_data['S'].append(torch.cat(all_complex_data['S'], dim=0))
        all_batch_data['Y'].append(torch.cat(all_complex_data['Y'], dim=0))
        all_batch_data['Y_t'].append(torch.cat(all_complex_data['Y_t'], dim=0))
        all_batch_data['Y_m'].append(torch.cat(all_complex_data['Y_m'], dim=0))
        all_batch_data['R_idx'].append(torch.cat(all_complex_data['R_idx'], dim=0))

        L_max = max(curr_l, L_max)
    
    # Pad and stack all data in everything together along batch dimension.
    output_dict = {}
    for k in all_batch_data.keys():
        padded_tensors = []
        if k in ['X', 'xyz_37']: # Handle 3D tensors.
            for x in all_batch_data[k]:
                x_pad = F.pad(x, (0, 0, 0, 0, 0, L_max - x.shape[0]), 'constant', value=0.0)
                padded_tensors.append(x_pad)

        if k in ['xyz_37_m']: # Handle 3D tensors.
            for x in all_batch_data[k]:
                x_pad = F.pad(x, (0, 0, 0, L_max - x.shape[0]), 'constant', value=0)
                padded_tensors.append(x_pad)

        if k in ['mask', 'chain_labels', 'S', 'chain_mask', 'R_idx']: # Handle 1D tensors.
            for x in all_batch_data[k]:
                mask_value = -1
                if k in ['mask', 'chain_mask']:
                    mask_value = 0
                if k in ['S']:
                    mask_value = 20
                x_pad = F.pad(x, (0, L_max - x.shape[0]), 'constant', value=mask_value)
                padded_tensors.append(x_pad)

        if k in ['Y']:
            for x in all_batch_data[k]:
                x_pad = F.pad(x, (0, 0, 0, Y_max - x.shape[0]), 'constant', value=0)
                padded_tensors.append(x_pad)     

        if k in ['Y_t', 'Y_m']:
            for x in all_batch_data[k]:
                x_pad = F.pad(x, (0, Y_max - x.shape[0]), 'constant', value=0)
                padded_tensors.append(x_pad)
        
        if len(padded_tensors) > 0:
            padded_tensors = torch.stack(padded_tensors)
            output_dict[k] = padded_tensors

    out = featurize(output_dict, cutoff_for_score=cutoff_for_score, use_atom_context=use_atom_context, atom_context_num=atom_context_num)
    out['batch_size'] = out['X'].shape[0]

    return out


def featurize(input_dict, cutoff_for_score, use_atom_context, atom_context_num):
    output_dict = {}
    mask = input_dict["mask"]
    Y = input_dict["Y"]
    Y_t = input_dict["Y_t"]
    Y_m = input_dict["Y_m"]
    N = input_dict["X"][:, :,  0, :]
    CA = input_dict["X"][:, :, 1, :]
    C = input_dict["X"][:, :, 2, :]
    b = CA - N
    c = C - CA
    a = torch.cross(b, c, axis=-1)
    CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
    Y, Y_t, Y_m, D_XY = batch_get_nearest_neighbours(
        CB, mask, Y, Y_t, Y_m, atom_context_num
    )
    mask_XY = (D_XY < cutoff_for_score) * mask * Y_m[:, :, 0]
    output_dict["mask_XY"] = mask_XY
    if "side_chain_mask" in list(input_dict):
        output_dict["side_chain_mask"] = input_dict["side_chain_mask"]
    output_dict["Y"] = Y
    output_dict["Y_t"] = Y_t
    output_dict["Y_m"] = Y_m

    if not use_atom_context:
        output_dict["Y_m"] = 0.0 * output_dict["Y_m"]

    output_dict["R_idx"] = input_dict['R_idx']
    output_dict["chain_labels"] = input_dict["chain_labels"]
    output_dict["S"] = input_dict["S"]
    output_dict["chain_mask"] = input_dict["chain_mask"]
    output_dict["mask"] = input_dict["mask"]
    output_dict["X"] = input_dict["X"]

    if "xyz_37" in list(input_dict):
        output_dict["xyz_37"] = input_dict["xyz_37"]
        output_dict["xyz_37_m"] = input_dict["xyz_37_m"]

    return output_dict