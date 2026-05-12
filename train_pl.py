import os
import shutil
import numpy as np
import wandb
import torch
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import hydra
from omegaconf import DictConfig

import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D
from models.pl_model import MoleculeTrainer

from graphbap.bapnet import BAPNet

import warnings
warnings.filterwarnings("ignore")


@hydra.main(config_path='./configs', config_name='config')
def main(cfg: DictConfig):
    seed_everything(cfg.train.seed)

    wandb_logger = misc.init_wandb(cfg)
    ckpt_dir, vis_dir = misc.create_folders(cfg)

    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(cfg.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    if cfg.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    subsets = get_dataset(
        config=cfg.data,
        transform=transform,
    )

    train_set, val_set, test_set = subsets['train'], subsets['valid'], subsets['test']
    print(f'Training: {len(train_set)} Validation: {len(val_set)} Testing: {len(test_set)}')

    collate_exclude_keys = ['ligand_nbh_list']
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=0,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys,
        pin_memory=False,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_set,
        cfg.train.val_batch_size,
        shuffle=False,
        num_workers=0,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys,
        pin_memory=False,
    )

    net_cond = BAPNet(ckpt_path=cfg.net_cond.ckpt_path,
                      hidden_nf=cfg.net_cond.hidden_dim).to('cuda')

    model = MoleculeTrainer(
        cfg,
        protein_featurizer,
        ligand_featurizer,
        net_cond,
        )

    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='{epoch}',
        save_top_k=5,
        monitor='validation/loss',
        mode='min',
        save_last=True,
        every_n_epochs=1,
    )

    early_stop_callback = EarlyStopping(
        monitor='validation/loss',
        patience=cfg.train.patience,
        verbose=True,
        mode='min'
    )

    # Use Tensor Cores
    torch.set_float32_matmul_precision('high')

    strategy = cfg.sys.strategy
    if cfg.sys.strategy == 'ddp' and getattr(cfg.sys, 'find_unused_parameters', False):
        strategy = DDPStrategy(find_unused_parameters=True)

    trainer = pl.Trainer(
        accelerator=cfg.sys.accelerator,
        strategy=strategy,
        devices=cfg.sys.devices,
        num_nodes=cfg.sys.num_nodes,
        max_epochs=cfg.train.max_iters,
        num_sanity_val_steps=0,
        gradient_clip_val=cfg.train.max_grad_norm,
        sync_batchnorm=True,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    main()
