'''
sample_data.py
===================================================================================================================
Two core functionalities:

project_data – Encodes graph data through the GNN backbone and projects the resulting embeddings into a
               lower-dimensional latent space using a configurable reduction technique (e.g. UMAP).

sample_data  – Partitions the projected dataset into training, validation, and test splits using configurable
               sampling strategies.

'''

from catalyst.src.ml.nn.gnn.models.alignn import (Encoder_generic,Encoder_atomic,
                                                  Processor, Decoder,PositiveScalarsDecoder, ALIGNN)
from catalyst.src.characterization.sodas.model.sodas import SODAS
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
===================================================================================================================
Core Functions
===================================================================================================================
'''

def project_data(cat):
    '''
    PROJECT DATA
    '''
    cat.parameters['io_dict']['projection_dir'] = os.path.join(cat.parameters['io_dict']['main_path'],'projection')
    if not os.path.exists(cat.parameters['io_dict']['projection_dir']):
        os.makedirs(cat.parameters['io_dict']['projection_dir'])
    graph_data = [torch.load(file_name) for file_name in
                      glob.glob(os.path.join(cat.parameters['io_dict']['data_dir'], '*'))[::2]]
    # read data and perform projections
    print('Performing graph projections...')
    print('Length of graph_data: ' + str(len(graph_data)))
    projected_data = None
    encoded_data = []
    gids = []
    y = []
    follow_batch = ['x_atm', 'x_bnd', 'x_ang'] if hasattr(graph_data[0], 'x_ang') else ['x_atm', 'x_bnd']
    for data in graph_data:
        gids.append(data.gid)
        y.append(data.y)
    loader = DataLoader(graph_data, batch_size=parameters['loader_dict']['batch_size'][0],
                        shuffle=False, follow_batch=follow_batch,
                        num_workers=cat.parameters['loader_dict']['num_workers'])
    encoded_data = cat.parameters['model_dict']['model'].generate_gnn_latent_space(parameters=cat.parameters,
                                                                                   loader=loader)
    encoded_data = np.array(encoded_data)
    print(encoded_data)
    print(encoded_data.shape)
    cat.parameters['model_dict']['model'].fit_preprocess(data=encoded_data)
    cat.parameters['model_dict']['model'].fit_dim_red(data=encoded_data)
    projected_data = cat.parameters['model_dict']['model'].project_data(data=encoded_data)
    stored_projections = dict(
            projections=projected_data,
            gids=gids
    )
    save_dictionary(os.path.join(cat.parameters['io_dict']['projection_dir'], 'projection_data.npy'),
                                 stored_projections)
    return graph_data, projected_data, y

def sample_data(cat,graph_data,projected_data,y):
    '''
    SAMPLE DATA
    '''
    cat.parameters['io_dict']['samples_dir'] = os.path.join(cat.parameters['io_dict']['main_path'],'samples')
    if not os.path.exists(cat.parameters['io_dict']['samples_dir']):
        os.makedirs(cat.parameters['io_dict']['samples_dir'])
    #start sampling
    rng = np.random.default_rng(seed=cat.parameters['sampling_dict']['sampling_seed'])
    # REMOVE TEST DATA
    test_idx, nontest_idx = sampling.run_sampling(projected_data,y=y,
                                                  sampling_type=cat.parameters['sampling_dict']['sampling_types'][0],
                                                  split=cat.parameters['sampling_dict']['split'][0], rng=rng,
                                                  params_group=cat.parameters['sampling_dict']['params_groups'][0])
    stored_test_data = dict(
            projections=[projected_data[index] for index in test_idx],
            gids=[graph_data[index].gid for index in test_idx]
    )
    projected_data = [projected_data[index] for index in nontest_idx]
    graph_data = [graph_data[index] for index in nontest_idx]
    save_dictionary(os.path.join(cat.parameters['io_dict']['samples_dir'], 'test_data.npy'), stored_test_data)
    # REMOVE TRAINING DATA
    cat.parameters['io_dict']['model_dir'] = cat.parameters['io_dict']['samples_dir']
    if os.path.isdir(cat.parameters['io_dict']['model_dir']):
        shutil.rmtree(cat.parameters['io_dict']['model_dir'])
    os.mkdir(cat.parameters['io_dict']['model_dir'])
    for iteration in range(cat.parameters['model_dict']['n_models']):
        cat.parameters['io_dict']['model_dir'] = None
        del cat.parameters['io_dict']['model_dir']
        cat.parameters['io_dict']['model_dir'] = os.path.join(cat.parameters['io_dict']['samples_dir'], 'model_samples',
                                                              str(iteration))
        if os.path.isdir(cat.parameters['io_dict']['model_dir']):
            shutil.rmtree(cat.parameters['io_dict']['model_dir'])
        os.makedirs(cat.parameters['io_dict']['model_dir'], exist_ok=True)
        # sample data and train model
        train_idx, valid_idx = sampling.run_sampling(projected_data,
                                                     sampling_type=cat.parameters['sampling_dict']['sampling_types'][2],
                                                     split=cat.parameters['sampling_dict']['split'][2], rng=rng,
                                                     params_group=cat.parameters['sampling_dict']['params_groups'][2])
        train_data = [graph_data[index].gid for index in train_idx]
        valid_data = [graph_data[index].gid for index in valid_idx]
        print('Using the remaining ', len(valid_data), ' for validation')
        partitioned_data = dict(
                training_projections=[projected_data[index] for index in train_idx],
                validation_projections=[projected_data[index] for index in valid_idx],
                training=train_data,
                validation=valid_data
        )
        save_dictionary(os.path.join(cat.parameters['io_dict']['model_dir'], 'train_valid_split.npy'), partitioned_data)
    del graph_data

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

    # SODAS model: GNN backbone + UMAP-based latent-space projector
    sodas_model = SODAS(
        mod=ALIGNN(
            encoder=Encoder_atomic(
                num_species=n_types, cutoff=cutoff, dim=projection_indim, act=nn.SiLU()
            ),
            processor=Processor(
                num_convs=n_convs, dim=projection_indim, conv_type='mesh', act=nn.SiLU()
            ),
            decoder=Decoder(
                in_dim=projection_indim, out_dim=projection_outdim, act=nn.SiLU()
            ),
        ),
        ls_mod=umap_.UMAP(n_neighbors=10, min_dist=0.1, n_components=2),
    )
    # ── Catalyst Set-up and Running───────────────────────────────────────────────────
    cat = Catalyst()
    cat.set_params(parameters)
    cat.set_model(sodas_model)
    raw_data, projections, y = project_data(cat)
    sample_data(cat, graph_data=raw_data, projected_data=projections, y=y)