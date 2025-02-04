import gampixpy
import os
import yaml
import numpy as np

class Config (dict):
    def __init__(self, config_filename):
        self.config_filename = config_filename

        self.parse_config()
        self.compute_derived_parameters()
        
    def parse_config(self):
        # parse a config (yaml) file and store values
        # internally as a dict
        with open(self.config_filename) as config_file:
            for key, value in yaml.load(config_file, Loader = yaml.FullLoader).items():
                self[key] = value
        
        return

class DetectorConfig (Config):
    def compute_derived_parameters(self):
        # compute any required parameters from
        # those specified in the YAML

        return

class PhysicsConfig (Config):
    def compute_derived_parameters(self):
        # compute drift velocity, diffusion model

        return

class ReadoutConfig (Config):
    def compute_derived_parameters(self):
        # compute any required parameters from
        # those specified in the YAML

        # need to handle the case where the pitch doesn't evenly divide the span of the anode
        self['n_pixels_x'] = int((self['anode']['x_range'][1] - self['anode']['x_range'][0])/self['pixels']['pitch'])
        self['n_pixels_y'] = int((self['anode']['y_range'][1] - self['anode']['y_range'][0])/self['pixels']['pitch'])

        self['n_tiles_x'] = int((self['anode']['x_range'][1] - self['anode']['x_range'][0])/self['coarse_tiles']['pitch'])
        self['n_tiles_y'] = int((self['anode']['y_range'][1] - self['anode']['y_range'][0])/self['coarse_tiles']['pitch'])

        self['tile_volume_edges'] = (np.linspace(self['anode']['x_range'][0],
                                                 self['anode']['x_range'][1],
                                                 self['n_tiles_x']+1),
                                     np.linspace(self['anode']['y_range'][0],
                                                 self['anode']['y_range'][1],
                                                 self['n_tiles_y']+1))
        return

default_detector_params = DetectorConfig(os.path.join(gampixpy.__path__[0],
                                                      'detector_config',
                                                      'default.yaml'))
default_physics_params = PhysicsConfig(os.path.join(gampixpy.__path__[0],
                                                    'physics_config',
                                                    'default.yaml'))
default_readout_params = ReadoutConfig(os.path.join(gampixpy.__path__[0],
                                                    'readout_config',
                                                    'default.yaml'))
