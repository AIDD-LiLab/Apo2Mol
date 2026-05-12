import argparse
import os
import shutil
import time
import pickle

import numpy as np
import torch
import torch.multiprocessing as mp
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from utils.evaluation import atom_num
from graphbap.bapnet import BAPNet

def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]
    return all_step_v

def worker_process(i, device, data,
                   num_batch, batch_size, num_samples,
                   model: ScorePosNet3D, pos_only, sample_num_atoms, center_pos_mode, net_cond, cond_dim, num_steps,
                   all_pred_protein_pos, all_protein_pos_rmsd, all_protein_pos_tmscore,
                   all_pred_ligand_pos, all_pred_v,
                   all_pred_protein_pos_traj, all_pred_ligand_pos_traj, all_pred_v_traj,
                   all_pred_v0_traj, all_pred_vt_traj, time_list):
    # Set the current CUDA device
    torch.cuda.set_device(device)
    model = model.to(device)

    n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)

    # Move all data to the correct device (specified by the argument)
    batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)
    net_cond = net_cond.to(device)

    t1 = time.time()
    with torch.no_grad():
        batch_protein = batch.protein_element_batch.to(device)  # Ensure all tensors are on the right device
        if sample_num_atoms == 'prior':
            pocket_size = atom_num.get_space_size(batch.protein_pos.detach().cpu().numpy())
            ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
            batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
        elif sample_num_atoms == 'range':
            ligand_num_atoms = list(range(i * batch_size + 1, i * batch_size + n_data + 1))
            batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
        elif sample_num_atoms == 'ref':
            batch_ligand = batch.ligand_element_batch.to(device)
            ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
        else:
            raise ValueError

        center_pos = scatter_mean(batch.protein_pos_holo.to(device), batch_protein, dim=0).to(device)  # Move to device
        batch_center_pos = center_pos[batch_ligand].to(device)  # Ensure it's on the correct device
        init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos).to(device)

        if pos_only:
            init_ligand_v = batch.ligand_atom_feature_full.to(device)
        else:
            uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
            init_ligand_v = log_sample_categorical(uniform_logits).to(device)

        r = model.sample_diffusion(
            data=batch,
            protein_pos_apo=batch.protein_pos.to(device),
            protein_pos_holo=batch.protein_pos_holo.to(device),
            protein_v=batch.protein_atom_feature.float().to(device),
            batch_protein=batch_protein.to(device),

            init_ligand_pos=init_ligand_pos.to(device),
            init_ligand_v=init_ligand_v.to(device),
            batch_ligand=batch_ligand.to(device),

            num_steps=num_steps,
            pos_only=pos_only,
            center_pos_mode=center_pos_mode,
            net_cond=net_cond,
            cond_dim=cond_dim
        )
        protein_pos, ligand_pos, ligand_v, protein_pos_traj, ligand_pos_traj, ligand_v_traj = \
            r['protein_pos'], r['ligand_pos'], r['v'], r['protein_pos_traj'], r['ligand_pos_traj'], r['v_traj']
        protein_pos_rmsd, protein_pos_tmscore = r['protein_pos_rmsd'], r['protein_pos_tmscore']
        ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
        ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
        ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
        all_pred_protein_pos += [protein_pos.cpu().numpy()]
        all_pred_ligand_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]
        all_protein_pos_rmsd += [rmsd.cpu().numpy() for rmsd in protein_pos_rmsd]
        all_protein_pos_tmscore += [tmscore.cpu().numpy() for tmscore in protein_pos_tmscore]

        all_step_protein_pos = [[] for _ in range(n_data)]
        for p in protein_pos_traj:
            p_array = p.cpu().numpy().astype(np.float64)
            for k in range(n_data):
                all_step_protein_pos[k].append(p_array)
        all_step_protein_pos = [np.stack(step_pos) for step_pos in all_step_protein_pos]
        all_pred_protein_pos_traj += [p for p in all_step_protein_pos]

        all_step_ligand_pos = [[] for _ in range(n_data)]
        for p in ligand_pos_traj:
            p_array = p.cpu().numpy().astype(np.float64)
            for k in range(n_data):
                all_step_ligand_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
        all_step_ligand_pos = [np.stack(step_pos) for step_pos in all_step_ligand_pos]
        all_pred_ligand_pos_traj += [p for p in all_step_ligand_pos]

        ligand_v_array = ligand_v.cpu().numpy()
        all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

        all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
        all_pred_v_traj += [v for v in all_step_v]

        if not pos_only:
            all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
            all_pred_v0_traj += [v for v in all_step_v0]
            all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
            all_pred_vt_traj += [v for v in all_step_vt]
    t2 = time.time()
    time_list.append(t2 - t1)

def sample_diffusion_ligand(
        model: ScorePosNet3D,
        data,
        num_samples,
        batch_size=16, devices=['cuda:0'],
        num_steps=None, pos_only=False, center_pos_mode='protein', sample_num_atoms='prior', net_cond=None, cond_dim=128):

    assert net_cond is not None

    # Use Manager to share data between processes
    manager = mp.Manager()
    all_pred_protein_pos, all_pred_ligand_pos, all_pred_v = manager.list(), manager.list(), manager.list()
    all_protein_pos_rmsd, all_protein_pos_tmscore = manager.list(), manager.list()
    all_pred_protein_pos_traj, all_pred_ligand_pos_traj, all_pred_v_traj = manager.list(), manager.list(), manager.list()
    all_pred_v0_traj, all_pred_vt_traj = manager.list(), manager.list()
    time_list = manager.list()

    num_batch = int(np.ceil(num_samples / batch_size))

    # Ensure 'spawn' is used as the start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # already set

    # Create parallel processes — use the passed devices parameter
    processes = []
    for i in range(num_batch):
        device = devices[i % len(devices)]  # Assign GPU in a round-robin fashion
        # print(f'Process {i} on {device}')
        p = mp.Process(target=worker_process, args=(i, device, data,
                                                    num_batch, batch_size, num_samples, model, pos_only, 
                                                    sample_num_atoms, center_pos_mode, net_cond, cond_dim, 
                                                    num_steps, all_pred_protein_pos, all_protein_pos_rmsd, all_protein_pos_tmscore,
                                                    all_pred_ligand_pos, all_pred_v,
                                                    all_pred_protein_pos_traj, all_pred_ligand_pos_traj,
                                                    all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    return list(all_pred_protein_pos), list(all_protein_pos_rmsd), list(all_protein_pos_tmscore), \
           list(all_pred_ligand_pos), list(all_pred_v), \
           list(all_pred_protein_pos_traj), list(all_pred_ligand_pos_traj), list(all_pred_v_traj), \
           list(all_pred_v0_traj), list(all_pred_vt_traj), list(time_list)


def gpu_worker(task_queue, result_queue, gpu_device, worker_cfg):
    """
    Module-level worker that processes data points on a specific GPU.
    (Must be at module level for pickle/spawn compatibility.)
    """
    import os
    import shutil

    torch.cuda.set_device(gpu_device)
    device = torch.device(gpu_device)

    # Unpack config
    train_config = worker_cfg['train_config']
    config = worker_cfg['config']
    args = worker_cfg['args']
    result_path = worker_cfg['result_path']

    # Rebuild featurizers to get feature dims
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = train_config.data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)

    # Load net_cond on this GPU
    net_cond = BAPNet(ckpt_path=train_config.net_cond.ckpt_path,
                       hidden_nf=train_config.net_cond.hidden_dim).to(device)

    # Load model on this GPU
    ckpt_local = torch.load(config.model.checkpoint, map_location=device)
    model = ScorePosNet3D(
        train_config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(device)
    model_ckpt_local = ckpt_local['state_dict']
    model_ckpt_local = {k.replace("model.", ""): v for k, v in model_ckpt_local.items() if k.startswith("model.")}
    model.load_state_dict(model_ckpt_local)
    print(f"[{gpu_device}] Model loaded successfully!")

    while not task_queue.empty():
        try:
            data_id, data = task_queue.get(timeout=1)
        except Exception:
            break

        print(f"[{gpu_device}] Processing data ID: {data_id}")

        pred_protein_pos, pred_protein_pos_rmsd, pred_protein_pos_tmscore, \
        pred_ligand_pos, pred_v, \
        pred_protein_pos_traj, pred_ligand_pos_traj, pred_v_traj, \
        pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
            model, data,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            devices=[gpu_device],
            num_steps=config.sample.num_steps,
            pos_only=config.sample.pos_only,
            center_pos_mode=config.sample.center_pos_mode,
            sample_num_atoms=config.sample.sample_num_atoms,
            net_cond=net_cond,
            cond_dim=train_config.model.cond_dim
        )
        result = {
            'data': data,
            'pred_protein_pos': pred_protein_pos,
            'pred_protein_pos_rmsd': pred_protein_pos_rmsd,
            'pred_protein_pos_tmscore': pred_protein_pos_tmscore,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_v,
            'pred_protein_pos_traj': pred_protein_pos_traj,
            'pred_ligand_ligand_pos_traj': pred_ligand_pos_traj,
            'pred_ligand_v_traj': pred_v_traj,
            'time': time_list,
        }
        torch.save(result, os.path.join(result_path, f'result_{data_id}.pt'))
        print(f"[{gpu_device}] Saved result_{data_id}.pt")
        result_queue.put(data_id)


if __name__ == '__main__':
    root_dir = '.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=root_dir+'/configs/sampling.yaml')
    parser.add_argument('--train_config', type=str, default=root_dir+'/configs/training.yaml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--result_path', type=str, default=root_dir+'/sampled_results/apo2mol-plinder')
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=478)
    args = parser.parse_args()

    logger = misc.get_logger('sampling')

    config = misc.load_config(args.config)
    train_config = misc.load_config(args.train_config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    logger.info(f"Training Config: {train_config}")

    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = train_config.data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    subsets = get_dataset(
        config=train_config.data,
        transform=transform
    )
    train_set, test_set = subsets['train'], subsets['test']
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')

    topk = train_config.data.topk_prompt
    num_test = len(test_set)
    print(f"Number of test data: {num_test}")

    # --- Clamp indices to valid range ---
    start_index = max(0, args.start_index)
    end_index = min(args.end_index, num_test - 1)
    if args.start_index != start_index or args.end_index != end_index:
        print(f"NOTE: adjusted indices from [{args.start_index}, {args.end_index}] to [{start_index}, {end_index}] (dataset size={num_test})")
    data_ids = list(range(start_index, end_index + 1))

    # --- Determine all available GPUs ---
    all_devices = ["cuda:{}".format(i) for i in range(torch.cuda.device_count())]
    print(f"Using {len(all_devices)} GPUs: {all_devices}")

    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))

    # --- Pack config for workers (must be picklable, no nested functions) ---
    worker_cfg = {
        'train_config': train_config,
        'config': config,
        'args': args,
        'result_path': result_path,
    }

    # --- Put (data_id, data) tuples into task queue ---
    mp.set_start_method('spawn', force=True)
    task_queue = mp.Queue()
    for data_id in data_ids:
        task_queue.put((data_id, test_set[data_id]))

    result_queue = mp.Queue()

    print(f"Launching {len(all_devices)} worker processes...")

    workers = []
    for gpu_id in all_devices:
        p = mp.Process(target=gpu_worker, args=(task_queue, result_queue, gpu_id, worker_cfg))
        p.start()
        workers.append(p)

    # Wait for all workers
    for p in workers:
        p.join()

    completed = []
    while not result_queue.empty():
        completed.append(result_queue.get())
    print(f"All tasks completed! ({len(completed)} results)")
