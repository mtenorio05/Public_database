'''
test_models.py
===================================================================================================================
Two core functionalities:

test_model – Model testing with y value predictions

generate_rankings  – Model testing with all feature score calculations

'''


from catalyst.src.ml.nn.gnn.models.alignn import (Encoder_generic,Encoder_atomic, Processor, Decoder,
                                                  PositiveScalarsDecoder, ALIGNN)
from catalyst.src.ml.inference import predict_external, test_non_intepretable_external, predict_interpretable
from catalyst.src.characterization.sodas.model.sodas import SODAS
from catalyst.src.io.io import load_dictionary, save_dictionary
from catalyst.src.observer.params import Catalyst
from catalyst.src.ml.utils.loss import MaxNpercent
from torch_geometric.loader import DataLoader
import torch.multiprocessing as mp
import torch as torch
from torch import nn
from pathlib import Path, PurePath
import matplotlib.pyplot as plt
import numpy as np
import shutil, glob, os
from umap import umap_


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

def test_model(cat):
    '''
    TEST MODEL
    '''
    cat.parameters['io_dict']['write_indv_pred'] = True
    cat.parameters['io_dict']['results_dir'] = None
    del cat.parameters['io_dict']['results_dir']
    cat.parameters['io_dict']['results_dir'] = os.path.join(cat.parameters['io_dict']['main_path'], 'predictions')
    if not os.path.isdir(cat.parameters['io_dict']['results_dir']):
        os.makedirs(cat.parameters['io_dict']['results_dir'], exist_ok=True)
    models = [os.path.join(cat.parameters['io_dict']['model_dir'],'0')]
    for i, model in enumerate(models):
        cat.parameters['io_dict']['results_dir'] = os.path.join(cat.parameters['io_dict']['main_path'],
                                                                'predictions',str(i))
        if os.path.isdir(cat.parameters['io_dict']['results_dir']):
            shutil.rmtree(cat.parameters['io_dict']['results_dir'])
        os.makedirs(cat.parameters['io_dict']['results_dir'], exist_ok=True)
        cat.parameters['io_dict']['loaded_model_name'] = None
        del cat.parameters['io_dict']['loaded_model_name']
        cat.parameters['io_dict']['loaded_model_name'] = glob.glob(os.path.join(model, 'model*'))[0]
        test_non_intepretable_external(cat,'all', rank=0)
    return

def generate_rankings(cat,interpret):
    cat.parameters['io_dict']['write_indv_pred'] = True
    cat.parameters['io_dict']['results_dir'] = None
    del cat.parameters['io_dict']['results_dir']
    cat.parameters['io_dict']['results_dir'] = os.path.join(cat.parameters['io_dict']['main_path'], 'rankings')
    if not os.path.isdir(cat.parameters['io_dict']['results_dir']):
        os.makedirs(cat.parameters['io_dict']['results_dir'], exist_ok=True)
    models = [os.path.join(cat.parameters['io_dict']['model_dir'],'0')]
    for i, model in enumerate(models):
        cat.parameters['io_dict']['results_dir'] = os.path.join(cat.parameters['io_dict']['main_path'],
                                                                'rankings',str(i))
        if os.path.isdir(cat.parameters['io_dict']['results_dir']):
            shutil.rmtree(cat.parameters['io_dict']['results_dir'])
        os.makedirs(cat.parameters['io_dict']['results_dir'], exist_ok=True)
        cat.parameters['io_dict']['loaded_model_name'] = None
        del cat.parameters['io_dict']['loaded_model_name']
        cat.parameters['io_dict']['loaded_model_name'] = glob.glob(os.path.join(model, 'model*'))[0]
        predict_external(cat, 'all', rank=0,interpretable=interpret)
    return

'''
=============================================================================
Entry point
=============================================================================
'''

if __name__ == '__main__':
    mp.set_start_method('spawn')
    # ── Architecture hyperparameters ──────────────────────────────────────────
    n_types          = 119   # fictitious type labels (one per periodic-table element)
    projection_indim = 100   # GNN hidden dimensionality for the SODAS encoder
    projection_outdim = 10   # latent-space output dimensionality after UMAP
    regression_indim  = 100  # GNN hidden dimensionality for the regression model
    regression_outdim = 10   # regression output dimensionality
    cutoff  = 3.5            # radial cutoff (Å) for graph edge construction
    n_convs = 5              # number of mesh-convolution layers

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
            batch_size=[1,1],
            num_workers=0,
            shuffle_steps=10
        ),
        # Training / model settings
        model_dict=dict(
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
    test_model(cat)
    cat.set_model(alignnd_model)
    generate_rankings(cat,1)




















