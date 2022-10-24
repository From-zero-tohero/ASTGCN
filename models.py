''' Define the ASTGCN model '''
import torch
from torch.nn import init

from ASTGCN.astgcn import ASTGCN
from ASTGCN.utils import norm_adj_matrix


def create_model(adj_filepath, points_per_hour, n_predictions, n_vertices):
    device = "cuda:0"
    device = torch.device(device)
    indus_A, share_A, concept_A = norm_adj_matrix(adj_filepath, device=device)

    # mixin = [dict(n_vertices=n_vertices, n_predictions=n_predictions, A=share_A)]

    mixin = [dict(n_vertices=n_vertices, n_predictions=n_predictions, A=indus_A),
             dict(n_vertices=n_vertices, n_predictions=n_predictions, A=share_A),
             dict(n_vertices=n_vertices, n_predictions=n_predictions, A=concept_A)]

    submodules = [{
        "blocks": [
            {
                "in_channels": 5,
                "in_timesteps": points_per_hour,
                'out_channels': 32,
                'gcn_filters': 32,
                'tcn_strides': 1
            },
            {
                "in_channels": 32,
                "in_timesteps": points_per_hour,
                'out_channels': 32,
                'gcn_filters': 32,
                'tcn_strides': 1,
            }
        ]
    }] * 3

    astgcn = ASTGCN(submodules=submodules, mixin=mixin).to(device)
    for name, params in astgcn.named_parameters(recurse=True):
        if params.dim() > 1:
            init.xavier_uniform_(params)
        else:
            init.uniform_(params)
    return astgcn
