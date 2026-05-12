import argparse
import os

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter
from Bio.PDB import PDBParser, PDBIO

from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length
from utils import misc, reconstruct, transforms
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask


def write_log(log_file, message):
    with open(log_file, 'a') as f:
        f.write(message + '\n')


def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')


def print_ring_ratio(all_ring_sizes, logger):
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    root_dir = '.'
    parser.add_argument('--sample_path', default=os.path.join(root_dir, './sampled_results/apo2mol-plinder'), type=str)
    parser.add_argument('--result_path', default=os.path.join(root_dir, './eval_results/apo2mol-plinder'), type=str)
    parser.add_argument('--pocket_type', default='gen', type=str) # 'apo', 'holo', and 'gen'
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_start_index', type=int, default=0)
    parser.add_argument('--eval_end_index', type=int, default=478)
    parser.add_argument('--save', type=eval, default=True)
    parser.add_argument('--protein_root', type=str, default='./apo2mol_dataset/data_folder')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking_mode', type=str, default='vina_score', choices=['qvina', 'vina_score', 'vina_dock', 'none'])
    parser.add_argument('--exhaustiveness', type=int, default=128)
    args = parser.parse_args()

    log_path = os.path.join(args.result_path, 'eval_v2.log')

    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    # Load generated data
    print(f'Load generated data from {args.sample_path}')
    write_log(log_path, f'Load generated data from {args.sample_path}')
    results_fn_list = glob(os.path.join(args.sample_path, '*result_*.pt'))
    results_fn_list = sorted(results_fn_list, key=lambda x: int(os.path.basename(x)[:-3].split('_')[-1]))
    eval_start_index = args.eval_start_index
    eval_end_index = args.eval_end_index
    if args.eval_start_index is None:
        eval_start_index = 0
    if args.eval_end_index is None:
        eval_start_index = len(results_fn_list) - 1

    results_fn_list = results_fn_list[eval_start_index: eval_end_index+1]
    num_examples = len(results_fn_list)
    logger.info(f'Load generated data done! sample_id[{eval_start_index}:{eval_end_index}] examples for evaluation.')
    write_log(log_path, f'Load generated data done! sample_id[{eval_start_index}:{eval_end_index}] examples for evaluation.')
    write_log(log_path, f'Number of generated data: {len(results_fn_list)}')
    write_log(log_path, f'Generated data: {results_fn_list}')

    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_recon_success, n_eval_success, n_complete = 0, 0, 0
    results = []
    all_pair_dist, all_bond_dist = [], []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()
    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):
        r = torch.load(r_name)  # ['data', 'pred_ligand_pos', 'pred_ligand_v', 'pred_ligand_pos_traj', 'pred_ligand_v_traj', 'rmsd']
        data = r['data']
        protein_filename = data.holo_filename
        ligand_filename = data.ligand_filename

        all_pred_ligand_pos = r['pred_ligand_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3]
        all_pred_ligand_v = r['pred_ligand_v_traj']
        all_pred_protein_pos = r['pred_protein_pos_traj']
        num_samples += len(all_pred_ligand_pos)
        best_vina = 100
        result = {}
        for sample_idx, (pred_pos, pred_v, pred_protein_pos) in enumerate(tqdm(zip(all_pred_ligand_pos, all_pred_ligand_v, all_pred_protein_pos), desc='Sample')):
            pred_pos, pred_v = pred_pos[args.eval_step], pred_v[args.eval_step]
            pred_protein_pos = pred_protein_pos[args.eval_step]

            # stability check
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)
            all_atom_types += Counter(pred_atom_type)
            r_stable = analyze.check_stability(pred_pos, pred_atom_type)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]

            pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
            all_pair_dist += pair_dist

            # reconstruction
            try:
                pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
                smiles = Chem.MolToSmiles(mol)
                if 'apo2mol' in args.protein_root:
                    # Assign the pred_protein_pos to the original pdb file
                    protein_file_name = "receptor_apo_pocket10.pdb"
                    protein_fn = os.path.join(
                        os.path.dirname(r['data'].ligand_filename),
                        protein_file_name
                    )
                    protein_path = os.path.join(args.protein_root, protein_fn)
                    # replace the original protein position with the predicted protein position in the protein_path pdb file
                    parser = PDBParser(QUIET=True)
                    structure = parser.get_structure('protein', protein_path)

                    pred_protein_pos_np = pred_protein_pos

                    atom_iter = structure.get_atoms()
                    for i, atom in enumerate(atom_iter):
                        if i < len(pred_protein_pos_np):
                            atom.set_coord(pred_protein_pos_np[i])
                        else:
                            break # Stop if predicted positions are fewer than atoms in PDB

                    id_name = r['data'].ligand_filename.replace('.sdf', '').split('/')[0]
                    # mkdir according to id_name
                    os.makedirs(os.path.join(result_path, id_name), exist_ok=True)
                    pred_protein_path = os.path.join(result_path, id_name, f"{example_idx}_{sample_idx}_{id_name}_protein_pred.pdb")

                    io = PDBIO()
                    io.set_structure(structure)
                    io.save(pred_protein_path)
            except reconstruct.MolReconsError:
                if args.verbose:
                    logger.warning('Reconstruct failed %s' % f'{example_idx}_{sample_idx}')
                continue
            n_recon_success += 1

            if '.' in smiles:
                continue
            n_complete += 1

            # chemical and docking check
            try:
                chem_results = scoring_func.get_chem(mol)
                if args.docking_mode == 'qvina':
                    vina_task = QVinaDockingTask.from_generated_mol(
                        mol, r['data'].ligand_filename, protein_root=args.protein_root)
                    vina_results = vina_task.run_sync()
                elif args.docking_mode in ['vina_score', 'vina_dock']:
                    vina_task = VinaDockingTask.from_generated_mol(
                        mol, r['data'].ligand_filename, pred_protein_path, protein_root=args.protein_root, pocket_type=args.pocket_type)

                    # save pdb file
                    if "apo2mol" in args.protein_root:
                        if args.pocket_type == 'holo':
                            protein_file_name = "receptor_holo_pocket10.pdb"
                            protein_fn = os.path.join(
                                os.path.dirname(r['data'].ligand_filename),
                                protein_file_name
                            )
                        elif args.pocket_type == 'apo':
                            protein_file_name = "receptor_apo_pocket10.pdb"
                            protein_fn = os.path.join(
                                os.path.dirname(r['data'].ligand_filename),
                                protein_file_name
                            )
                        elif args.pocket_type == 'gen':
                            protein_file_name = "receptor_holo_pocket10.pdb"
                            protein_fn = os.path.join(
                                os.path.dirname(r['data'].ligand_filename),
                                protein_file_name
                            )
                        protein_path = os.path.join(args.protein_root, protein_fn)
                        origin_ligand_fn = r['data'].ligand_filename
                        origin_ligand_path = os.path.join(args.protein_root, origin_ligand_fn)
                        id_name = origin_ligand_fn.replace('.sdf', '').split('/')[0]
                        # save protein pdb file with mol in the save folder with the name of example_idx_sample_idx_protein.pdb
                        os.system(f'cp {protein_path} {os.path.join(result_path, id_name, f"{example_idx}_{id_name}_holo.pdb")}')
                        Chem.MolToPDBFile(mol, os.path.join(result_path, id_name, f'{example_idx}_{sample_idx}_{id_name}.pdb'))
                        os.system(f'cp {origin_ligand_path} {os.path.join(result_path, id_name, f"{example_idx}_{id_name}_ligand.sdf")}')

                    # score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
                    minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
                    vina_results = {
                        # 'score_only': score_only_results,
                        'minimize': minimize_results
                    }
                    if minimize_results[0]['affinity'] < best_vina:
                        result = {
                            'mol': mol,
                            'smiles': smiles,
                            'ligand_filename': r['data'].ligand_filename,
                            'pred_pos': pred_pos,
                            'pred_v': pred_v,
                            'chem_results': chem_results,
                            'vina': vina_results,
                            'example_idx': example_idx,
                            'sample_idx': sample_idx,
                            # 'rmsd': rmsd_value,
                        }
                        best_vina = minimize_results[0]['affinity']
                    if args.docking_mode == 'vina_dock':
                        docking_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness)
                        vina_results['dock'] = docking_results
                else:
                    vina_results = None

                n_eval_success += 1
            except:
                if args.verbose:
                    logger.warning('Evaluation failed for %s' % f'{example_idx}_{sample_idx}')
                continue

        if result:
            results.append(result)
            bond_dist = eval_bond_length.bond_distance_from_mol(result['mol'])
            all_bond_dist += bond_dist
            success_pair_dist += pair_dist
            success_atom_types += Counter(pred_atom_type)
            # break
    logger.info(f'Evaluate done! {num_samples} samples in total.')

    fraction_mol_stable = all_mol_stable / num_samples
    fraction_atm_stable = all_atom_stable / all_n_atom
    fraction_recon = n_recon_success / num_samples
    fraction_eval = n_eval_success / num_samples
    fraction_complete = n_complete / num_samples
    validity_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
        'recon_success': fraction_recon,
        'eval_success': fraction_eval,
        'complete': fraction_complete
    }
    print_dict(validity_dict, logger)

    c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
    logger.info('JS bond distances of complete mols: ')
    print_dict(c_bond_length_dict, logger)

    success_pair_length_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)
    success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_length_profile)
    print_dict(success_js_metrics, logger)

    atom_type_js = eval_atom_type.eval_atom_type_distribution(success_atom_types)
    logger.info('Atom type JS: %.4f' % atom_type_js)

    if args.save:
        eval_bond_length.plot_distance_hist(success_pair_length_profile,
                                            metrics=success_js_metrics,
                                            save_path=os.path.join(result_path, f'pair_dist_hist_{eval_start_index}-to-{eval_end_index}.png'))

    logger.info('Number of reconstructed mols: %d, complete mols: %d, evaluated mols: %d' % (
        n_recon_success, n_complete, len(results)))

    qed = [r['chem_results']['qed'] for r in results]
    sa = [r['chem_results']['sa'] for r in results]
    logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))
    logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))
    if args.docking_mode == 'qvina':
        vina = [r['vina'][0]['affinity'] for r in results]
        logger.info('Vina:  Mean: %.3f Median: %.3f' % (np.mean(vina), np.median(vina)))
    elif args.docking_mode in ['vina_dock', 'vina_score']:
        # vina_score_only = [r['vina']['score_only'][0]['affinity'] for r in results]
        vina_min = [r['vina']['minimize'][0]['affinity'] for r in results]
        # print("vina_min: ", vina_min)
        # logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
        logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
        if args.docking_mode == 'vina_dock':
            vina_dock = [r['vina']['dock'][0]['affinity'] for r in results]
            logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))

    # check ring distribution
    print_ring_ratio([r['chem_results']['ring_size'] for r in results], logger)

    if args.save:
        torch.save({
            'stability': validity_dict,
            'bond_length': all_bond_dist,
            'all_results': results
        }, os.path.join(result_path, f'metrics_{args.eval_step}_{eval_start_index}-to-{eval_end_index}.pt'))
