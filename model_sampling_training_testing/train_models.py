'''
train_models.py
===================================================================================================================
One core functionality:

train_model – Performs the model training using a predefined or user-defined model.

'''

from catalyst.src.ml.nn.gnn.models.alignn import (Encoder_generic,Encoder_atomic, Processor,
                                                  Decoder,PositiveScalarsDecoder, ALIGNN)
from catalyst.src.ml.training import run_training
from catalyst.src.graph.generic_build import generic_graph_gen
from catalyst.src.ml.utils.distributed import cuda_destroy
import catalyst.src.utilities.sampling as sampling
from catalyst.src.io.io import load_dictionary, save_dictionary
from catalyst.src.observer.params import Catalyst
from catalyst.src.ml.utils.loss import MaxNpercent
from torch_geometric.loader import DataLoader
import torch.multiprocessing as mp
import torch as torch
from torch import nn
from pathlib import Path, PurePath
import os


'''
Global parameter initialization
'''
global n_types
global projection_indim
global projection_outdim
global cutoff
global regression_indim
global regression_outdim
global n_convs
global n_data
global n_nodes
global n_dim

'''
Function definitions
'''

def train_model(cat):
    '''
    PERFORM MODEL TRAINING
    '''
    cat.parameters['io_dict']['samples_dir'] = None
    del cat.parameters['io_dict']['samples_dir']
    for iteration in range(cat.parameters['model_dict']['n_models']):
        cat.parameters['io_dict']['samples_dir'] = (
                                         os.path.join(cat.parameters['io_dict']['main_path'],'samples','model_samples'))
        if cat.parameters['device_dict']['run_ddp']:
            print('Performing training on model ', iteration)
            processes = []
            for rank in range(cat.parameters['device_dict']['world_size']):
                p = mp.Process(target=run_training, args=(rank, iteration, cat,))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            cuda_destroy()
        else:
            run_training(rank=0, iteration=iteration, cat=cat)
    return

'''
=============================================================================
Entry point
=============================================================================
'''

if __name__ == '__main__':
    mp.set_start_method('spawn')
    # ── Architecture hyperparameters ──────────────────────────────────────────
    n_types = 119  # fictitious type labels (one per periodic-table element)
    projection_indim = 100  # GNN hidden dimensionality for the SODAS encoder
    projection_outdim = 10  # latent-space output dimensionality after UMAP
    regression_indim = 100  # GNN hidden dimensionality for the regression model
    regression_outdim = 10  # regression output dimensionality
    cutoff = 3.5  # radial cutoff (Å) for graph edge construction
    n_convs = 5  # number of mesh-convolution layers

    # ── Experiment parameter dictionary (for Catalyst) ───────────────────────────────────────
    parameters = dict(
        # Hardware / distributed-training settings
        device_dict=dict(
            world_size=2,
            device='cuda',
            ddp_backend='gloo',
            run_ddp=False,
            pin_memory=False,
            find_unused_parameters=False
        ),
        # File-system paths
        io_dict=dict(
            main_path=str(Path(__file__).parent),
            loaded_model_name=None,
            data_dir='',
            model_dir=None,
            results_dir=None,
            samples_dir=None,
            projection_dir=None,
            remove_old_model=True,
            write_indv_pred=False,
            graph_read_format=0
        ),
        # Sampling strategy configuration
        # Index 0 → test split, index 2 → train/valid split
        sampling_dict=dict(
            sampling_types=['y_bin', 'gaussian_mixture','gaussian_mixture'],
            split=[0.2, 0.2, 0.9],
            sampling_seed=112358,
            params_groups=[{
                'clusters': 5,
            }, {
                'clusters': 5,
            }, {
                'clusters': 5,
            }]
        ),
        # DataLoader settings
        loader_dict=dict(
            shuffle_loader=False,
            batch_size=[5,5],
            num_workers=0,
            shuffle_steps=10
        ),
        # Training / model settings
        model_dict=dict(
            #Number of model being trained
            n_models=1,
            num_epochs=1500,
            train_delta=0.001,
            train_tolerance=0.001,
            max_deltas=10,
            loss_params={
                'function':'MaxNpercent',
                'sub_function':torch.nn.L1Loss(),
                'percent':0.1
            },
            accumulate_loss='exact',
            model=None,
            interpretable=False,
            pre_training=False,
            restart_training=False,
            optimizer_params=dict(
                dynamic_lr=False,
                optimizer='AdamW',
                params_group={
                    'lr': 0.0001
                }
            )
        )
    )

    # ── Model Initialization───────────────────────────────────────────────────
    # Regression model: GNN backbone with a positive-scalar output head
    alignnd_model = ALIGNN(
                        encoder=Encoder_atomic(num_species=n_types, cutoff=cutoff, dim=regression_indim, act=nn.SiLU()),
                        processor=Processor(num_convs=n_convs, dim=regression_indim, conv_type='mesh', act=nn.SiLU()),
                        decoder=PositiveScalarsDecoder(dim=regression_indim, act=nn.SiLU()),
                    )

    # ── Catalyst Set-up and Running───────────────────────────────────────────────────
    cat = Catalyst()
    cat.set_params(parameters)
    cat.set_model(alignnd_model)
    train_model(cat)





















