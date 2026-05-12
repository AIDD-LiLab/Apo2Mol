import logging
import os
import random
import time

import numpy as np
import torch
import yaml
import wandb
import hydra
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.loggers import WandbLogger
from easydict import EasyDict


class BlackHole(object):
    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


def init_wandb(args: DictConfig):
    if 'SLURM_JOB_ID' in os.environ:
        if int(os.environ.get('SLURM_PROCID', 0)) == 0:
            wandb.login(key=args.wandb.wandb_key)
            wandb.init(
                project=args.wandb.wandb_project,
                name=args.wandb.wandb_task,
                config=OmegaConf.to_container(args, resolve=True),
                entity=args.wandb.wandb_entity,
                mode=args.wandb.wandb_status,
            )
            wandb_logger = WandbLogger(project=args.wandb.wandb_project,
                                    log_model=False,
                                    offline=False)
            wandb_logger = [wandb_logger]
        else:
            wandb_logger = []
    else:
        if int(os.environ.get('LOCAL_RANK', 0)) == 0:
            wandb.login(key=args.wandb.wandb_key)
            wandb.init(
                project=args.wandb.wandb_project,
                name=args.wandb.wandb_task,
                config=OmegaConf.to_container(args, resolve=True),
                entity=args.wandb.wandb_entity,
                mode=args.wandb.wandb_status,
            )
            wandb_logger = WandbLogger(project=args.wandb.wandb_project,
                                    log_model=False,
                                    offline=False)
            wandb_logger = [wandb_logger]
        else:
            wandb_logger = []

    return wandb_logger


def create_folders(cfg: DictConfig):
    ckpt_dir = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'checkpoints')
    vis_dir = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'vis')

    try:
        os.makedirs(ckpt_dir)
    except OSError:
        pass
    try:
        os.makedirs(vis_dir)
    except OSError:
        pass

    return ckpt_dir, vis_dir


def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_new_log_dir(root='./logs', prefix='', tag=''):
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_hyperparams(writer, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k: v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)


def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):
    return tuple(argstr.split(','))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
