import os
import pickle
import lmdb
import multiprocessing as mp
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from utils.data import PDBProtein, parse_sdf_file, compute_residue_transforms
from .pl_data import ProteinLigandData, ApoHoloLigandData, torchify_dict

def process_item(item):
    i, (holo_pocket_fn, apo_pocket_fn, ligand_fn, *_), raw_path = item
    if holo_pocket_fn is None or apo_pocket_fn is None:
        return i, None, ligand_fn
    try:
        data_prefix = raw_path
        holo_pocket_dict = PDBProtein(os.path.join(data_prefix, holo_pocket_fn)).to_dict_atom()
        apo_pocket_dict = PDBProtein(os.path.join(data_prefix, apo_pocket_fn)).to_dict_atom()
        ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
        data = ApoHoloLigandData.from_apo_holo_ligand_dicts(
            apo_dict=torchify_dict(apo_pocket_dict),
            holo_dict=torchify_dict(holo_pocket_dict),
            ligand_dict=torchify_dict(ligand_dict),
        )
        data.protein_rotations, _, data.protein_translations, \
        data.protein_chi_apo, data.protein_chi_holo, data.protein_chi_mask = \
            compute_residue_transforms(
                protein_pos_apo=data.protein_pos,
                protein_pos_holo=data.protein_pos_holo,
                protein_atom_name=data.protein_atom_name,
                protein_atom_to_aa_name=data.protein_atom_to_aa_name,
                protein_atom_to_aa_group=data.protein_atom_to_aa_group,
            )
        assert data.protein_rotations.size(0) == max(data.protein_atom_to_aa_group) + 1
        assert data.protein_translations.size(0) == max(data.protein_atom_to_aa_group) + 1
        assert data.protein_chi_apo.size(0) == max(data.protein_atom_to_aa_group) + 1
        assert data.protein_chi_holo.size(0) == max(data.protein_atom_to_aa_group) + 1
        assert data.protein_chi_mask.size(0) == max(data.protein_atom_to_aa_group) + 1
        # if data.protein_rotations.size(0) <= 5:
        #     return i, None, ligand_fn
        if 0 in data.ligand_bond_type:
            return i, None, ligand_fn

        data.apo_filename = apo_pocket_fn
        data.holo_filename = holo_pocket_fn
        data.ligand_filename = ligand_fn
        data = data.to_dict()  # avoid torch_geometric version issue
        return i, pickle.dumps(data), None
    except Exception:
        return i, None, ligand_fn

class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, index_path, pocket_type, transform=None, version='final'):
        super().__init__()
        self.raw_path = raw_path.rstrip('/') # /blue/yanjun.li/pfq7pm.virginia/AIDD/ApoDiffusion/crossdock/crossdocked_v1.1_rmsd1.0
        self.pocket_type = pocket_type
        self.index_path = index_path
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + "_" + pocket_type + f'_apo2mol_{version}.lmdb')
        self.transform = transform
        self.db = None

        self.keys = None

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _process(self):
        print("Processing data")
        db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            tasks = [(i, item, self.raw_path) for i, item in enumerate(index) if item[0] is not None and item[1] is not None]

            pool = mp.Pool(processes=mp.cpu_count()//2)

            with tqdm(total=len(tasks)) as pbar:
                for i, data_bytes, ligand_fn in pool.imap_unordered(process_item, tasks):
                    if data_bytes is None:
                        num_skipped += 1
                        if ligand_fn:
                            print('Error processing entry (%d)' % (i,))
                            print("Ligand file:", ligand_fn)
                    else:
                        txn.put(
                            key=str(i).encode(),
                            value=data_bytes
                        )
                    pbar.update()
            pool.close()
            pool.join()

        db.close()

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.get_ori_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db()
        try:
            # key = self.keys[idx]
            key = str(idx).encode()
        except:
            print('error idx:', idx)
            print('len(self.keys):', len(self.keys))
        data = pickle.loads(self.db.begin().get(key))
        # data = ProteinLigandData(**data)
        data = ApoHoloLigandData(**data)
        data.id = idx
        assert data.protein_pos.size(0) > 0
        return data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    dataset = PocketLigandPairDataset(args.path)
    print(len(dataset), dataset[0])
