import tqdm
import numpy as np
import torch
import torch.sparse as sp
import h5py
import os

import gampixpy
from gampixpy import config

def tiles_to_tensor(tile_hits):
    tile_coords = torch.tensor(np.array([tile_hits['tile x'],
                                         tile_hits['tile y'],
                                         tile_hits['hit z']]))
    tile_feats = torch.tensor(np.array([tile_hits['hit charge'],
                                        ]))

    return tile_coords, tile_feats

def pixels_to_tensor(pixel_hits):
    pixel_coords = torch.tensor(np.array([pixel_hits['pixel x'],
                                          pixel_hits['pixel y'],
                                          pixel_hits['hit z']]))
    pixel_feats = torch.tensor(np.array([pixel_hits['hit charge'],
                                         ]))
    
    return pixel_coords, pixel_feats

def plot_coord_tensor(pixel_index_tensor):

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(*np.array(pixel_index_tensor))
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'y')
    ax.set_zlabel(r'z')
    plt.show()

def tile_coords_to_indices(tile_coords_tensor, readout_config, origin = 'edge'):
    spacing = torch.tensor([readout_config['coarse_tiles']['pitch'],
                            readout_config['coarse_tiles']['pitch'],
                            readout_config['coarse_tiles']['z_bin_width'],
                            ])

    if origin == 'edge': # this mode sets the lowest-value coordinate in each direction to 0
        min_voxel = np.array([readout_config['anode']['x_range'][0],
                              readout_config['anode']['y_range'][0],
                              readout_config['anode']['z_range'][0],
                              ])
        tile_index_tensor = torch.div(tile_coords_tensor - min_voxel[:,None],
                                      spacing[:,None],
                                      rounding_mode = 'trunc').long()
        if torch.any(tile_index_tensor):
            tile_index_tensor -= torch.min(tile_index_tensor, dim = 1).values[:,None]
            
    elif origin == 'coordinate': # this mode sets the voxel closest to the origin in coordinate space to (0,0,0)
        tile_index_tensor = torch.div(tile_coords_tensor,
                                      spacing[:,None],
                                      rounding_mode = 'trunc').long()
        if torch.any(tile_index_tensor):
            drift_zero = torch.min(tile_index_tensor, dim = 1).values[2,None]
            origin = torch.tensor([[0, 0, drift_zero]]).T
            tile_index_tensor -= origin
    
    return tile_index_tensor

def pixel_coords_to_indices(pixel_coords_tensor, readout_config, origin = 'edge'):
    spacing = torch.tensor([readout_config['pixels']['pitch'],
                            readout_config['pixels']['pitch'],
                            readout_config['pixels']['z_bin_width'],
                            ])

    # valid options for setting the origin:
    #   'edge'
    #   'coordinate'
    # ...more to come?
    if origin == 'edge': # this mode sets the lowest-value coordinate in each direction to 0
        min_voxel = np.array([readout_config['anode']['x_range'][0],
                              readout_config['anode']['y_range'][0],
                              readout_config['anode']['z_range'][0],
                              ])

        pixel_index_tensor = torch.div(pixel_coords_tensor - min_voxel[:,None],
                                       spacing[:,None],
                                       rounding_mode = 'trunc').long()
        if torch.any(pixel_index_tensor):
            pixel_index_tensor -= torch.min(pixel_index_tensor, dim = 1).values[:,None]
    if origin == 'coordinate': # this mode sets the voxel closest to the origin in coordinate space to (0,0,0)
        pixel_index_tensor = torch.div(pixel_coords_tensor,
                                       spacing[:,None],
                                       rounding_mode = 'trunc').long()
        if torch.any(pixel_index_tensor):
            drift_zero = torch.min(pixel_index_tensor, dim = 1).values[2,None]
            origin = torch.tensor([[0, 0, drift_zero]]).T
            pixel_index_tensor -= origin
        
    return pixel_index_tensor

def tensor_to_sparsetensor(coords, feats):
    st = torch.sparse_coo_tensor(coords, feats.T)
    
    return st

def get_event_hits(readout_data, event_id):
    coarse_hits_mask = readout_data['coarse_hits']['event id'] == event_id
    event_coarse_hits = readout_data['coarse_hits'][coarse_hits_mask]

    pixel_hits_mask = readout_data['pixel_hits']['event id'] == event_id
    event_pixel_hits = readout_data['pixel_hits'][pixel_hits_mask]

    return event_coarse_hits, event_pixel_hits

def make_event_sparsetensors(readout_data, event_id, readout_config = config.default_readout_params, **kwargs):
    event_tile_hits, event_pixel_hits = get_event_hits(readout_data, event_id)

    tile_coords_tensor, tile_charge_tensor = tiles_to_tensor(event_tile_hits)
    tile_index_tensor = tile_coords_to_indices(tile_coords_tensor, readout_config, **kwargs)

    tile_st = tensor_to_sparsetensor(tile_index_tensor, tile_charge_tensor)

    pixel_coords_tensor, pixel_charge_tensor = pixels_to_tensor(event_pixel_hits)
    pixel_index_tensor = pixel_coords_to_indices(pixel_coords_tensor, readout_config, **kwargs)

    pixel_st = tensor_to_sparsetensor(pixel_index_tensor, pixel_charge_tensor)

    return tile_st, pixel_st

def get_event_coo_tensors(readout_data, event_id, readout_config = config.default_readout_params, **kwargs):
    event_tile_hits, event_pixel_hits = get_event_hits(readout_data, event_id)

    tile_coords_tensor, tile_charge_tensor = tiles_to_tensor(event_tile_hits)
    tile_index_tensor = tile_coords_to_indices(tile_coords_tensor, readout_config, **kwargs).T

    # tile_st = tensor_to_sparsetensor(tile_index_tensor, tile_charge_tensor)

    pixel_coords_tensor, pixel_charge_tensor = pixels_to_tensor(event_pixel_hits)
    pixel_index_tensor = pixel_coords_to_indices(pixel_coords_tensor, readout_config, **kwargs).T

    # pixel_st = tensor_to_sparsetensor(pixel_index_tensor, pixel_charge_tensor)

    return (tile_index_tensor, tile_charge_tensor), (pixel_index_tensor, pixel_charge_tensor)

def get_event_meta(readout_data, event_id):
    event_meta_mask = readout_data['meta']['event id'] == event_id
    event_meta = readout_data['meta'][event_meta_mask]

    return event_meta

def main(args):
    gampixD_readout_config = config.ReadoutConfig(os.path.join(gampixpy.__path__[0],
                                                               'readout_config',
                                                               'GAMPixD.yaml'))

    gampix_readout_data = h5py.File(args.gampix_readout_file, 'r')

    for event_id in tqdm.tqdm(np.unique(gampix_readout_data['coarse_hits']['event id'])):
        print (event_id)

        pixel_st, tile_st = make_event_sparsetensors(gampix_readout_data,
                                                     event_id,
                                                     readout_config = gampixD_readout_config)

        print (torch.nn.Conv3d(1, 2, 3)(pixel_st))
        
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('gampix_readout_file',
                        type = str,
                        help = 'input file from which to read a simulated event readout')
    parser.add_argument('-o', '--output_file',
                        type = str,
                        default = "",
                        help = 'output hdf5 file to store coarse tile and pixel measurements')

    args = parser.parse_args()

    main(args)
